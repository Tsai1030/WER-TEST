"""Round 4：喚醒詞與講完判斷的自動測試（plan.md 第 15 章）。

- A 誤喚醒：Common Voice 英文句首尾相接成一條連續串流（模擬喚醒後的對話、旁人說話），
  加上 DEMAND 噪音源全段，計算每小時誤喚醒次數。
- B 講完判斷：調參集的句子組成「喚醒後說話」的情境（可搶話、句中停頓、背景噪音），
  逐幀機率算一次後套用多組參數，計算截斷率與講完後的等待時間。
- C 正確喚醒：使用者自錄的「小咪小咪」（data/wake/positive/）與音近詞（hard_negative/）。

輸出到 output_dir：kws_negative.csv、kws_negative_hits.csv、endpoint.csv、kws_positive.csv。
已存在的輸出檔會跳過（刪除該檔即可重跑）。
"""

import itertools
import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from library.data import (
    get_duration,
    load_commonvoice_tsv,
    load_manifest,
    manifest_source_files,
    read_audio,
    speakers_of,
)
from library.degrade import add_noise, noise_bank, prevent_clipping
from library.endpoint import SAMPLE_RATE, Endpointer, decide, endpoint_params
from library.kws import KeywordSpotter
from library.methods import get_method
from library.utils.paths import resolve_path
from library.utils.yaml_io import load_yaml

log = logging.getLogger(__name__)

NEGATIVE_FILE = "kws_negative.csv"
NEGATIVE_HITS_FILE = "kws_negative_hits.csv"
ENDPOINT_FILE = "endpoint.csv"
POSITIVE_FILE = "kws_positive.csv"
WAKE_KINDS = ("positive", "hard_negative")


@dataclass
class WakeConfig:
    """Round 4 設定（configs/round4.yaml）。各區段保留為 dict，欄位說明見設定檔註解。"""

    name: str
    seed: int
    output_dir: Path
    wake_dir: Path
    kws: dict
    negatives: dict = field(default_factory=dict)
    endpoint: dict = field(default_factory=dict)
    endpoint_sim: dict = field(default_factory=dict)
    positive: dict = field(default_factory=dict)
    live: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)


def load_wake_config(path: str | Path) -> WakeConfig:
    """讀取並驗證 Round 4 設定。

    例外:
        FileNotFoundError: 設定檔不存在。
        ValueError: 缺少必要區段、kws 模型或門檻設定錯誤、endpoint 參數不合法。
    """
    raw = load_yaml(resolve_path(path))
    for key in ("name", "paths", "kws"):
        if key not in raw:
            raise ValueError(f"設定檔缺少 {key}")
    kws = raw["kws"]
    if not kws.get("models"):
        raise ValueError("kws.models 不可為空")
    for t in kws.get("thresholds", [0.25]):
        if not 0 < float(t) < 1:
            raise ValueError(f"kws.thresholds 必須在 (0, 1)：{t}")
    cfg = WakeConfig(
        name=raw["name"],
        seed=int(raw.get("seed", 42)),
        output_dir=resolve_path(raw["paths"]["output_dir"]),
        wake_dir=resolve_path(raw["paths"].get("wake_dir", "data/wake")),
        kws=kws,
        **{k: raw.get(k) or {} for k in ("negatives", "endpoint", "endpoint_sim", "positive", "live", "record")},
    )
    for _, params in endpoint_variants(cfg):
        endpoint_params(params)
    live_model = cfg.live.get("kws_model")
    if live_model and live_model not in kws["models"]:
        raise ValueError(f"live.kws_model {live_model!r} 不在 kws.models 中")
    return cfg


def spotter_config(cfg: WakeConfig, model: str, threshold: float) -> dict:
    """組合單一 KeywordSpotter 的設定（kws 共用欄位 + 模型檔 + 門檻）。

    例外:
        KeyError: model 不在 kws.models 中。
    """
    shared = {k: v for k, v in cfg.kws.items() if k not in ("models", "thresholds")}
    return {**shared, **cfg.kws["models"][model], "keywords_threshold": float(threshold)}


def kws_variants(cfg: WakeConfig) -> list[tuple[str, float]]:
    """回傳要測的 (喚醒詞模型, 門檻) 組合，依設定檔順序。"""
    return [(m, float(t)) for m in cfg.kws["models"] for t in cfg.kws.get("thresholds", [0.25])]


def endpoint_variants(cfg: WakeConfig) -> list[tuple[str, dict]]:
    """把 endpoint.sweep 與 endpoint.default 合併展開，回傳 [(名稱, 參數), ...]。

    名稱格式為 t<threshold>_s<min_silence_ms>（例：t0.5_s800）；sweep 為空時只有 default。
    """
    default = dict(cfg.endpoint.get("default") or {})
    sweep = cfg.endpoint.get("sweep") or {}
    keys = list(sweep)
    combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(sweep[k] for k in keys))] or [{}]
    out = []
    for c in combos:
        p = endpoint_params({**default, **c})
        out.append((f"t{p['threshold']}_s{int(p['min_silence_ms'])}", {**default, **c}))
    return out


def _load_16k(path: Path) -> np.ndarray:
    import librosa

    wav, sr = read_audio(path)
    return wav if sr == SAMPLE_RATE else librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)


def _load_spotters(cfg: WakeConfig) -> dict[tuple[str, float], KeywordSpotter]:
    spotters = {}
    for model, t in kws_variants(cfg):
        s = KeywordSpotter(spotter_config(cfg, model, t))
        s.load()
        spotters[(model, t)] = s
    return spotters


def negative_clips(cfg: WakeConfig) -> pd.DataFrame:
    """依 seed 抽出誤喚醒測試用的 Common Voice 句子（排除測試集與調參集的說話者），湊滿 negatives.hours。

    回傳:
        DataFrame：path、sentence、gender、audio_sec（不含 client_id）。

    例外:
        FileNotFoundError: TSV、clips 或 exclude manifest 不存在。
        ValueError: 可用的句子不足 hours。
    """
    neg = cfg.negatives
    frames = [load_commonvoice_tsv(resolve_path(d) / neg.get("tsv_name", "test.tsv"), resolve_path(d) / "clips")
              for d in neg["raw"].values()]
    df = pd.concat(frames, ignore_index=True)
    excluded: set[str] = set()
    for m in neg.get("exclude_manifests", []):
        excluded |= speakers_of(df, manifest_source_files(m))
    df = df[~df["client_id"].isin(excluded)].sort_values("path", kind="stable").reset_index(drop=True)
    df = df.iloc[np.random.default_rng(cfg.seed).permutation(len(df))]
    need, total, rows = float(neg.get("hours", 1.0)) * 3600, 0.0, []
    for r in df.to_dict("records"):
        sec = get_duration(Path(r["path"]))
        rows.append({"path": r["path"], "sentence": r["sentence"], "gender": r["gender"], "audio_sec": sec})
        total += sec
        if total >= need:
            return pd.DataFrame(rows)
    raise ValueError(f"可用的句子只有 {total / 3600:.2f} 小時，不足 {need / 3600:.2f} 小時")


def _feed_all(streams: dict, wav: np.ndarray, step: int) -> dict:
    hits = {k: [] for k in streams}
    for i in range(0, len(wav), step):
        chunk = wav[i : i + step]
        for k, s in streams.items():
            hits[k] += s.accept(chunk)
    return hits


def run_kws_negative(cfg: WakeConfig) -> Path:
    """A：在連續的英文語音與噪音串流上跑所有 (模型, 門檻)，記錄每次（誤）喚醒。

    英文串流與噪音串流各自從頭建立一條 KWS 串流，句與句之間不重設（模擬一直開著的麥克風）。

    回傳:
        kws_negative.csv 路徑（每列：model、threshold、source、audio_sec、n_false）；
        逐次喚醒另存 kws_negative_hits.csv（item 為句子編號，sentence 為原文）。
    """
    out = cfg.output_dir
    path = out / NEGATIVE_FILE
    if path.is_file():
        log.info("已存在，跳過：%s", path)
        return path
    out.mkdir(parents=True, exist_ok=True)
    neg = cfg.negatives
    clips = negative_clips(cfg)
    gap = np.zeros(round(float(neg.get("gap_sec", 0.3)) * SAMPLE_RATE), dtype=np.float32)
    step = round(SAMPLE_RATE * float(cfg.kws.get("chunk_ms", 100)) / 1000)
    noise_dir = resolve_path(neg["noise_source"])
    noise_items = [(n, noise_dir / f"{n}.wav") for n in neg.get("noise_files", [])]
    spotters = _load_spotters(cfg)
    rows, hit_rows = [], []
    try:
        sources = [("cv_en", [(i, Path(p), s) for i, (p, s) in enumerate(zip(clips["path"], clips["sentence"], strict=True))]),
                   ("noise", [(n, p, "") for n, p in noise_items])]
        for source, items in sources:
            if not items:
                continue
            streams = {k: s.stream(SAMPLE_RATE) for k, s in spotters.items()}
            starts: list[float] = []  # 每個 item 在串流中的起點（秒）
            t = 0.0
            for item, p, sentence in tqdm(items, desc=f"kws {source}", unit="段", dynamic_ncols=True):
                wav = np.concatenate([_load_16k(p), gap])
                starts.append(t)
                for k, hits in _feed_all(streams, wav, step).items():
                    for h in hits:
                        j = max(0, bisect_right(starts, h.t_sec) - 1)
                        hit_rows.append({"model": k[0], "threshold": k[1], "source": source, "item": items[j][0],
                                         "t_sec": round(h.t_sec - starts[j], 2), "keyword": h.keyword,
                                         "sentence": items[j][2]})
                t += len(wav) / SAMPLE_RATE
            for k in spotters:
                n = sum(1 for h in hit_rows if (h["model"], h["threshold"], h["source"]) == (*k, source))
                rows.append({"model": k[0], "threshold": k[1], "source": source, "audio_sec": round(t, 1), "n_false": n})
    finally:
        for s in spotters.values():
            s.unload()
    pd.DataFrame(hit_rows, columns=["model", "threshold", "source", "item", "t_sec", "keyword", "sentence"]).to_csv(
        out / NEGATIVE_HITS_FILE, index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


@dataclass(frozen=True)
class Episode:
    """一次「喚醒後說話」的模擬：wav 從開始收音起算，ref_start / ref_end 為實際語音範圍（秒）。"""

    episode: str
    utt_id: str
    utt_id2: str
    lead_sec: float
    pause_sec: float | None
    background: str
    wav: np.ndarray
    ref_start: float
    ref_end: float


def _trim_to_speech(vad: object, wav: np.ndarray, pad: int) -> np.ndarray:
    segs = vad.segments(wav, SAMPLE_RATE)
    if not segs:
        return wav
    return wav[max(0, segs[0][0] - pad) : min(len(wav), segs[-1][1] + pad)]


def _mix_background(wav: np.ndarray, speech: np.ndarray, noise: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """疊加背景噪音：噪音功率依「語音段」的功率決定（整段含靜音，不能用整段平均）；起點由 seed 決定。"""
    start = int(np.random.default_rng(seed).integers(len(noise)))
    seg = noise[(start + np.arange(len(wav))) % len(noise)].astype(np.float64)
    p_speech = float(np.mean(speech.astype(np.float64) ** 2))
    scale = np.sqrt(p_speech / (float(np.mean(seg**2)) * 10 ** (snr_db / 10)))
    return prevent_clipping((wav + scale * seg).astype(np.float32))


def build_episodes(cfg: WakeConfig) -> list[Episode]:
    """依 endpoint_sim 組出所有模擬情境：句子 × 開口延遲 × 停頓 × 背景。

    兩句相接時第二句取下一句（最後一句接第一句）。每句先切到語音範圍並加 trim_pad_ms，
    停頓為兩句語音之間的實際間隔（pad 算在停頓內）；ref_start / ref_end 不含 pad。噪音起點由 seed 決定。

    例外:
        FileNotFoundError: manifest 或噪音源不存在。
    """
    sim = cfg.endpoint_sim
    utts = [u for u in load_manifest(resolve_path(sim["manifest"])) if u.utt_id.startswith("utt")]  # 不含 noise 列
    pad = round(float(sim.get("trim_pad_ms", 50)) / 1000 * SAMPLE_RATE)
    vad = get_method("vad", {"vad": {"impl": "silero", "threshold": 0.5, "min_speech_ms": 250,
                                     "min_silence_ms": 300, "pad_ms": 0}})
    vad.load("cpu")
    try:
        speech = [_trim_to_speech(vad, _load_16k(u.wav_path), pad) for u in utts]
    finally:
        vad.unload()
    bank = {name: mix for name, mix, _, _ in noise_bank(sim)}
    names = sorted(bank)
    tail = np.zeros(round(float(sim.get("tail_sec", 4.0)) * SAMPLE_RATE), dtype=np.float32)
    rng = np.random.default_rng(cfg.seed)
    out = []
    n = len(utts)
    for i, lead, pause in itertools.product(range(n), sim.get("lead_sec", [0.0]), sim.get("pauses_sec", [None])):
        parts = [np.zeros(round(float(lead) * SAMPLE_RATE), dtype=np.float32), speech[i]]
        j = i
        if pause is not None:
            j = (i + 1) % n
            gap = max(0, round(float(pause) * SAMPLE_RATE) - 2 * pad)  # 停頓 = 兩句語音之間的實際間隔（含 pad）
            parts += [np.zeros(gap, dtype=np.float32), speech[j]]
        body = np.concatenate(parts)
        ref_start = float(lead) + pad / SAMPLE_RATE
        ref_end = (len(body) - pad) / SAMPLE_RATE
        clean = np.concatenate([body, tail])
        voiced = np.concatenate([speech[i], speech[j]]) if pause is not None else speech[i]
        noise_name = names[int(rng.integers(len(names)))]
        seed = int(rng.integers(2**31))
        for bg, snr in (sim.get("backgrounds") or {"none": None}).items():
            wav = clean if snr is None else _mix_background(clean, voiced, bank[noise_name], float(snr), seed)
            ep_id = f"{utts[i].utt_id}_l{lead}_p{pause}_{bg}"
            out.append(Episode(ep_id, utts[i].utt_id, utts[j].utt_id if pause is not None else "", float(lead),
                               None if pause is None else float(pause), bg, wav, ref_start, ref_end))
    return out


def run_endpoint_sim(cfg: WakeConfig) -> Path:
    """B：對每個模擬情境計算一次逐幀機率，套用所有講完判斷參數，逐列寫入 endpoint.csv。

    每列欄位：episode、utt_id、utt_id2、lead_sec、pause_sec、background、variant、threshold、min_silence_ms、
    reason、end_sec、speech_start、speech_end、ref_start、ref_end、truncated（語音結束前就判定講完）、
    delay_sec（講完到判定的等待時間，未截斷才有）、cut_sec（被截掉的語音秒數）。
    """
    out = cfg.output_dir
    path = out / ENDPOINT_FILE
    if path.is_file():
        log.info("已存在，跳過：%s", path)
        return path
    out.mkdir(parents=True, exist_ok=True)
    episodes = build_episodes(cfg)
    variants = endpoint_variants(cfg)
    ep = Endpointer()
    ep.load()
    rows = []
    try:
        for e in tqdm(episodes, desc="endpoint", unit="段", dynamic_ncols=True):
            probs = ep.probs(e.wav)
            for name, params in variants:
                p = endpoint_params(params)
                r = decide(probs, params)
                truncated = r.reason == "silence" and r.end_sec < e.ref_end
                rows.append({
                    "episode": e.episode, "utt_id": e.utt_id, "utt_id2": e.utt_id2, "lead_sec": e.lead_sec,
                    "pause_sec": e.pause_sec, "background": e.background, "variant": name,
                    "threshold": p["threshold"], "min_silence_ms": p["min_silence_ms"], "reason": r.reason,
                    "end_sec": round(r.end_sec, 3),
                    "speech_start": None if r.speech_start_sec is None else round(r.speech_start_sec, 3),
                    "speech_end": None if r.speech_end_sec is None else round(r.speech_end_sec, 3),
                    "ref_start": round(e.ref_start, 3), "ref_end": round(e.ref_end, 3), "truncated": truncated,
                    "delay_sec": round(r.end_sec - e.ref_end, 3) if r.reason == "silence" and not truncated else None,
                    "cut_sec": round(max(0.0, e.ref_end - r.end_sec), 3) if r.reason != "no_speech" else round(e.ref_end - e.ref_start, 3),
                })
    finally:
        ep.unload()
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def wake_recordings(cfg: WakeConfig) -> list[tuple[str, str, Path]]:
    """回傳自錄音檔 [(kind, style, path), ...]；style 取自檔名 <style>_<nn>.wav。"""
    out = []
    for kind in WAKE_KINDS:
        for p in sorted((cfg.wake_dir / kind).glob("*.wav")):
            out.append((kind, p.stem.rsplit("_", 1)[0], p))
    return out


def run_kws_positive(cfg: WakeConfig) -> Path | None:
    """C：在自錄音檔上跑所有 (模型, 門檻)；正樣本另疊加 positive.snrs 的噪音。沒有錄音時回傳 None。

    回傳:
        kws_positive.csv 路徑（每列：model、threshold、kind、style、file、snr、n_detect、first_t）。
    """
    recs = wake_recordings(cfg)
    if not recs:
        log.info("沒有自錄音檔（%s），略過正確喚醒測試；請先執行 07_record_wake.py", cfg.wake_dir)
        return None
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / POSITIVE_FILE
    sim = cfg.endpoint_sim
    bank = noise_bank(sim) if sim else ()
    spotters = _load_spotters(cfg)
    rows = []
    try:
        for n, (kind, style, p) in enumerate(tqdm(recs, desc="kws positive", unit="檔", dynamic_ncols=True)):
            wav = _load_16k(p)
            snrs = cfg.positive.get("snrs", [None]) if kind == "positive" else [None]
            for snr in snrs:
                x = wav if snr is None else prevent_clipping(add_noise(wav, bank[n % len(bank)][1], float(snr), cfg.seed + n))
                for (model, t), s in spotters.items():
                    hits = s.spot(x, SAMPLE_RATE)
                    rows.append({"model": model, "threshold": t, "kind": kind, "style": style, "file": p.name,
                                 "snr": "clean" if snr is None else f"snr{int(snr):02d}".replace("-", "m"),
                                 "n_detect": len(hits), "first_t": round(hits[0].t_sec, 2) if hits else None})
    finally:
        for s in spotters.values():
            s.unload()
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def summarize_kws(out_dir: Path) -> dict[str, pd.DataFrame]:
    """彙總喚醒詞結果。

    回傳:
        {"negative": 每個 (model, threshold) 各來源的每小時誤喚醒次數,
         "positive": 正確喚醒率（%，依 snr）與音近詞誤喚醒次數（有錄音時才有）}。
    """
    res = {}
    neg_path = out_dir / NEGATIVE_FILE
    if neg_path.is_file():
        neg = pd.read_csv(neg_path, encoding="utf-8")
        neg["fa_per_hour"] = neg["n_false"] / (neg["audio_sec"] / 3600)
        res["negative"] = neg.pivot_table(index=["model", "threshold"], columns="source",
                                          values=["n_false", "fa_per_hour"], sort=False)
    pos_path = out_dir / POSITIVE_FILE
    if pos_path.is_file():
        pos = pd.read_csv(pos_path, encoding="utf-8")
        pos["hit"] = pos["n_detect"] > 0
        p = pos[pos["kind"] == "positive"].pivot_table(index=["model", "threshold"], columns="snr", values="hit",
                                                       aggfunc="mean", sort=False) * 100
        p = p.add_prefix("recall_")
        hn = pos[pos["kind"] == "hard_negative"]
        if len(hn):
            p["hard_negative_false"] = hn.groupby(["model", "threshold"])["n_detect"].apply(lambda s: int((s > 0).sum()))
            p["hard_negative_files"] = hn.groupby(["model", "threshold"]).size()
        res["positive"] = p
    return res


def summarize_endpoint(path: Path, max_pause: float = 0.6, backgrounds: tuple[str, ...] = ("snr30", "snr10"),
                       max_truncated_pct: float = 5.0) -> pd.DataFrame:
    """彙總講完判斷模擬，並依事先訂定的規則推薦參數。

    推薦規則（plan.md 15.4）：在「單句或停頓 ≤ max_pause 秒」且背景為 backgrounds 的情境中，
    截斷率與失控率（max_length / eof）都 ≤ max_truncated_pct 的參數裡，取等待時間中位數最小者。

    回傳:
        DataFrame（index 為 variant）：各停頓的截斷率、各背景的失控率 / 漏收率、等待時間中位數與 p90、recommended。
    """
    df = pd.read_csv(path, encoding="utf-8")
    df["pause"] = df["pause_sec"].map(lambda v: "single" if pd.isna(v) else f"p{v:.1f}")
    df["runaway"] = df["reason"].isin(["max_length", "eof"])
    df["missed"] = df["reason"] == "no_speech"
    order = list(dict.fromkeys(df["variant"]))
    g = df.groupby("variant", sort=False)
    out = pd.DataFrame(index=order)
    out["threshold"] = g["threshold"].first()
    out["min_silence_ms"] = g["min_silence_ms"].first()
    trunc = df.pivot_table(index="variant", columns="pause", values="truncated", aggfunc="mean") * 100
    out = out.join(trunc.add_prefix("trunc_"))
    out = out.join((df.pivot_table(index="variant", columns="background", values="runaway", aggfunc="mean") * 100).add_prefix("runaway_"))
    out = out.join((df.pivot_table(index="variant", columns="background", values="missed", aggfunc="mean") * 100).add_prefix("missed_"))
    out["delay_med"] = g["delay_sec"].median()
    out["delay_p90"] = g["delay_sec"].quantile(0.9)
    scope = df[(df["pause_sec"].isna() | (df["pause_sec"] <= max_pause)) & df["background"].isin(backgrounds)]
    sg = scope.groupby("variant", sort=False)
    out["scope_trunc"] = sg["truncated"].mean() * 100
    out["scope_runaway"] = sg["runaway"].mean() * 100
    out["scope_delay_med"] = sg["delay_sec"].median()
    ok = (out["scope_trunc"] <= max_truncated_pct) & (out["scope_runaway"] <= max_truncated_pct)
    out["recommended"] = False
    if ok.any():
        out.loc[out.loc[ok, "scope_delay_med"].idxmin(), "recommended"] = True
    return out
