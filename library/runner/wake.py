"""Round 4：喚醒詞與講完判斷的自動測試（plan.md 第 15 章）。

- A 誤喚醒：Common Voice 英文句首尾相接成一條連續串流（模擬喚醒後的對話、旁人說話），
  加上 DEMAND 噪音源全段，計算每小時誤喚醒次數。
- B 講完判斷：調參集的句子組成「喚醒後說話」的情境（可搶話、句中停頓、背景噪音），
  逐幀機率算一次後套用多組參數，計算截斷率與講完後的等待時間。
- C 正確喚醒：使用者自錄的「小咪小咪」（data/wake/positive/）與音近詞（hard_negative/）。
- D 整段流程：測試集的句子組成同樣的情境，講完判斷後把收到的音訊送入 ASR，
  計算 WER（與不受截斷影響的 oracle 比較）與「講完到出字」的時間。
- E 待機成本：喚醒詞偵測、VAD 的即時 CPU 佔用與每塊處理時間，各元件的常駐記憶體。

輸出到 output_dir：kws_negative.csv、kws_negative_hits.csv、endpoint.csv、kws_positive.csv、
pipeline.csv、standby.csv。
已存在的輸出檔會跳過（刪除該檔即可重跑）。
"""

import itertools
import logging
import multiprocessing
import time
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
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
from library.endpoint import FRAME_SAMPLES, SAMPLE_RATE, Endpointer, decide, endpoint_params
from library.kws import KeywordSpotter
from library.methods import get_method
from library.models import get_model
from library.monitor import CpuTimer, Timer, current_rss_mb
from library.scoring import align, compare_per_utterance, paired_bootstrap
from library.textnorm import get_normalizer
from library.utils.paths import resolve_path
from library.utils.yaml_io import load_yaml

log = logging.getLogger(__name__)

NEGATIVE_FILE = "kws_negative.csv"
NEGATIVE_HITS_FILE = "kws_negative_hits.csv"
ENDPOINT_FILE = "endpoint.csv"
POSITIVE_FILE = "kws_positive.csv"
PIPELINE_FILE = "pipeline.csv"
STANDBY_FILE = "standby.csv"
ORACLE = "oracle"  # D 的對照：在實際語音結束 + trim_pad_ms 處切
_SIM_KEYS = ("manifest", "lead_sec", "pauses_sec", "backgrounds")  # pipeline 可覆寫的 endpoint_sim 欄位
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
    pipeline: dict = field(default_factory=dict)
    standby: dict = field(default_factory=dict)


def load_wake_config(path: str | Path) -> WakeConfig:
    """讀取並驗證 Round 4 設定。

    例外:
        FileNotFoundError: 設定檔不存在。
        ValueError: 缺少必要區段、kws 模型或門檻設定錯誤、endpoint 參數不合法、
            pipeline.variants 不在講完判斷參數中。
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
        **{k: raw.get(k) or {} for k in ("negatives", "endpoint", "endpoint_sim", "positive", "live", "record",
                                          "pipeline", "standby")},
    )
    names = [name for name, _ in endpoint_variants(cfg)]  # endpoint_variants 已驗證參數
    for v in cfg.pipeline.get("variants", []):
        if v != ORACLE and v not in names:
            raise ValueError(f"pipeline.variants {v!r} 不在講完判斷參數中（可用：{names}）")
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


def build_episodes(cfg: WakeConfig, sim: dict | None = None) -> list[Episode]:
    """依 sim（預設為 endpoint_sim）組出所有模擬情境：句子 × 開口延遲 × 停頓 × 背景。

    兩句相接時第二句取下一句（最後一句接第一句）。每句先切到語音範圍並加 trim_pad_ms，
    停頓為兩句語音之間的實際間隔（pad 算在停頓內）；ref_start / ref_end 不含 pad。噪音起點由 seed 決定。

    例外:
        FileNotFoundError: manifest 或噪音源不存在。
    """
    sim = cfg.endpoint_sim if sim is None else sim
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


def run_pipeline(cfg: WakeConfig) -> Path:
    """D：整段流程模擬（講完判斷 → 辨識），在測試集上驗證 B 選出的參數。

    情境的組法與 B 相同（pipeline 可覆寫 manifest、lead_sec、pauses_sec、backgrounds）。每個情境算一次逐幀機率，
    依 pipeline.variants 判定講完，把 [0, end_sec) 送入 ASR（與 09_live_demo 相同）；
    oracle 在實際語音結束 + trim_pad_ms 處切，作為不受截斷影響的對照。沒有偵測到說話（no_speech）時不辨識。

    回傳:
        pipeline.csv 路徑（每列：episode、utt_id、utt_id2、pause_sec、background、variant、reason、truncated、
        end_sec、ref_end、audio_sec、wait_sec（判定時間 − 語音結束）、asr_sec、response_sec（講完到出字，
        只有正常判定講完才有）、ref_raw、hyp_raw、ref_norm、hyp_norm、wer、S、D、I、N）。

    例外:
        KeyError: 設定缺少 pipeline.asr_model。
    """
    out = cfg.output_dir
    path = out / PIPELINE_FILE
    if path.is_file():
        log.info("已存在，跳過：%s", path)
        return path
    out.mkdir(parents=True, exist_ok=True)
    pl = cfg.pipeline
    sim = {**cfg.endpoint_sim, **{k: pl[k] for k in _SIM_KEYS if k in pl}}
    episodes = build_episodes(cfg, sim)
    refs = {u.utt_id: u.ref_raw for u in load_manifest(resolve_path(sim["manifest"]))}
    variants = dict(endpoint_variants(cfg))
    names = pl.get("variants") or [ORACLE]
    pad = float(sim.get("trim_pad_ms", 50)) / 1000
    normalizer = get_normalizer(pl.get("normalizer", "english"))
    ep = Endpointer()
    ep.load()
    asr = get_model(pl["asr_model"])
    asr.load(pl.get("device", "cpu"), "fp32", int(pl.get("num_threads", 4)))
    rows = []
    try:
        asr.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)  # 暖機：第一次呼叫較慢，不計時
        for e in tqdm(episodes, desc="pipeline", unit="段", dynamic_ncols=True):
            probs = ep.probs(e.wav)
            ref_raw = refs[e.utt_id] + (f" {refs[e.utt_id2]}" if e.utt_id2 else "")
            ref_norm = normalizer(ref_raw)
            for name in names:
                if name == ORACLE:
                    reason, end = ORACLE, e.ref_end + pad
                else:
                    r = decide(probs, variants[name])
                    reason, end = r.reason, r.end_sec
                audio = e.wav[: round(end * SAMPLE_RATE)]
                hyp, asr_sec = "", 0.0
                if reason != "no_speech":
                    with Timer() as t:
                        hyp = asr.transcribe(audio, SAMPLE_RATE).text
                    asr_sec = t.ms / 1000
                truncated = reason == "silence" and end < e.ref_end
                wait = end - e.ref_end
                hyp_norm = normalizer(hyp)
                a = align(ref_norm, hyp_norm)
                rows.append({
                    "episode": e.episode, "utt_id": e.utt_id, "utt_id2": e.utt_id2, "pause_sec": e.pause_sec,
                    "background": e.background, "variant": name, "reason": reason, "truncated": truncated,
                    "end_sec": round(end, 3), "ref_end": round(e.ref_end, 3),
                    "audio_sec": round(len(audio) / SAMPLE_RATE, 3), "wait_sec": round(wait, 3),
                    "asr_sec": round(asr_sec, 3),
                    "response_sec": round(wait + asr_sec, 3) if reason == "silence" and not truncated else None,
                    "ref_raw": ref_raw, "hyp_raw": hyp, "ref_norm": ref_norm, "hyp_norm": hyp_norm,
                    "wer": a.wer, "S": a.S, "D": a.D, "I": a.I, "N": a.N,
                })
    finally:
        asr.unload()
        ep.unload()
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def summarize_pipeline(path: Path, base: str = ORACLE, seed: int = 12345) -> pd.DataFrame:
    """彙總整段流程模擬：各 variant 的 WER、與 base 的 ΔWER（以情境為單位配對 bootstrap）、截斷率、
    講完到出字的時間。

    回傳:
        DataFrame（index 為 variant）：wer（%）、各停頓的 wer_<pause>、d_wer、ci_lo、ci_hi（百分點）、
        improved / worse / same（情境數）、trunc_pct、wait_med、asr_med、response_med、response_p90（秒）、
        audio_sec_mean。base 本身的 Δ 欄位為空。

    例外:
        ValueError: 檔案中沒有 base。
    """
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False, na_values=[""])
    if base not in set(df["variant"]):
        raise ValueError(f"{path.name} 中沒有 {base}")
    df["errors"] = df["S"] + df["D"] + df["I"]
    df["pause"] = df["pause_sec"].map(lambda v: "single" if pd.isna(v) else f"p{v:.1f}")
    ref = df[df["variant"] == base].set_index("episode")
    rows = {}
    for name, g in df.groupby("variant", sort=False):
        g = g.set_index("episode").loc[ref.index]
        row = {"wer": g["errors"].sum() / g["N"].sum() * 100}
        for pause, gp in g.groupby("pause", sort=True):
            row[f"wer_{pause}"] = gp["errors"].sum() / gp["N"].sum() * 100
        if name != base:
            be, me, n = ref["errors"].tolist(), g["errors"].tolist(), ref["N"].tolist()
            row["d_wer"] = row["wer"] - ref["errors"].sum() / ref["N"].sum() * 100
            row["ci_lo"], row["ci_hi"] = paired_bootstrap(be, me, n, seed=seed)
            row.update(compare_per_utterance(be, me))
        row["trunc_pct"] = g["truncated"].astype(bool).mean() * 100
        row["wait_med"] = g.loc[g["reason"] == "silence", "wait_sec"].median()
        row["asr_med"] = g["asr_sec"].median()
        row["response_med"] = g["response_sec"].median()
        row["response_p90"] = g["response_sec"].quantile(0.9)
        row["audio_sec_mean"] = g["audio_sec"].mean()
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def _concat_audio(paths: list[Path], sec: float) -> np.ndarray:
    """依序讀入並首尾相接，取前 sec 秒（不夠時從頭重複）。"""
    need = round(sec * SAMPLE_RATE)
    parts, total = [], 0
    while total < need:
        for p in paths:
            parts.append(_load_16k(p))
            total += len(parts[-1])
            if total >= need:
                break
    return np.concatenate(parts)[:need]


def _measure_component(cfg: WakeConfig, component: str, model: str) -> list[dict]:
    """在目前行程中載入一個元件並量測（由 run_standby 在獨立子行程中呼叫）。"""
    sb = cfg.standby
    sec = float(sb.get("audio_sec", 300))
    audios = {}
    if component != "asr":
        utts = [u for u in load_manifest(resolve_path(sb["manifest"])) if u.utt_id.startswith("utt")]
        noise_dir = resolve_path(cfg.negatives["noise_source"])
        audios = {"speech": _concat_audio([u.wav_path for u in utts], sec),
                  "noise": _concat_audio([noise_dir / f"{n}.wav" for n in cfg.negatives["noise_files"]], sec)}
    rss0 = current_rss_mb()
    threads, unit = None, 0
    if component == "kws":
        obj = KeywordSpotter(spotter_config(cfg, model, float(sb.get("threshold", 0.25))))
        threads = int(cfg.kws.get("num_threads", 1))
        unit = round(SAMPLE_RATE * float(cfg.kws.get("chunk_ms", 100)) / 1000)
        obj.load()
    elif component == "vad":
        obj = Endpointer()
        unit = FRAME_SAMPLES
        obj.load()
    else:
        obj = get_model(model)
        threads = int(sb.get("num_threads", 4))
        obj.load("cpu", "fp32", threads)
    rss1 = current_rss_mb()
    base = {"component": component, "model": model, "num_threads": threads, "rss_before_mb": round(rss0, 1),
            "rss_after_mb": round(rss1, 1), "ram_load_mb": round(rss1 - rss0, 1)}
    rows = []
    try:
        if not audios:
            return [{**base, "audio": None}]
        for name, wav in audios.items():
            if component == "kws":
                step = obj.stream(SAMPLE_RATE).accept
            else:
                obj.reset()
                step = obj.frame_prob
            times = []
            with CpuTimer() as ct:
                for i in range(0, len(wav) - unit + 1, unit):
                    t0 = time.perf_counter()
                    step(wav[i : i + unit])
                    times.append((time.perf_counter() - t0) * 1000)
            audio_sec = len(times) * unit / SAMPLE_RATE
            rows.append({**base, "audio": name, "audio_sec": round(audio_sec, 1),
                         "unit_ms": round(unit / SAMPLE_RATE * 1000, 1),
                         "proc_ms_med": round(float(np.median(times)), 3),
                         "proc_ms_p99": round(float(np.percentile(times, 99)), 3),
                         "cpu_pct": round(ct.cpu_sec / audio_sec * 100, 2), "rtf": round(ct.wall_sec / audio_sec, 4)})
    finally:
        obj.unload()
    return rows


def standby_components(cfg: WakeConfig) -> list[tuple[str, str]]:
    """回傳待機成本要量的 [(元件, 模型), ...]：每個喚醒詞模型、Silero VAD、ASR 模型（standby.asr_model）。"""
    asr = cfg.standby.get("asr_model")
    return [("kws", m) for m in cfg.kws["models"]] + [("vad", "silero")] + ([("asr", asr)] if asr else [])


def run_standby(cfg: WakeConfig) -> Path:
    """E：待機成本。每個元件在獨立的子行程中載入（記憶體互不影響），量測載入後增加的 RSS；
    喚醒詞與 VAD 另以即時的單位（KWS 每 chunk_ms、VAD 每 32 ms 一幀）送入 standby.audio_sec 秒的
    語音與噪音，量測每塊處理時間與 CPU 佔用。ASR 只量常駐記憶體（辨識時間見 D）。

    回傳:
        standby.csv 路徑（每列：component、model、num_threads、rss_before_mb、rss_after_mb、ram_load_mb、
        audio、audio_sec、unit_ms、proc_ms_med、proc_ms_p99、cpu_pct（佔一顆邏輯核心的 %）、rtf）。
    """
    out = cfg.output_dir
    path = out / STANDBY_FILE
    if path.is_file():
        log.info("已存在，跳過：%s", path)
        return path
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    ctx = multiprocessing.get_context("spawn")
    for component, model in tqdm(standby_components(cfg), desc="standby", unit="個", dynamic_ncols=True):
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            rows += pool.submit(_measure_component, cfg, component, model).result()
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path
