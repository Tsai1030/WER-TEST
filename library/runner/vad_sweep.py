"""Round 3：在調參集上掃描 VAD 參數 / 實作（只用一個模型），選出送到測試集驗證的設定。

調參集與測試集的說話者不重疊，純噪音片段取自噪音源的疊加用區段（測試集的 noise_only 取自保留區段），
避免在測試集上挑參數。輸出 sweep.csv（逐句），彙總由 summarize_sweep() 計算。
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from library.data import read_audio
from library.methods import get_method
from library.models import get_model
from library.runner.config import ExperimentConfig
from library.runner.matrix import degraded_path, split_utterances
from library.scoring import align
from library.textnorm import get_normalizer

log = logging.getLogger(__name__)

SWEEP_FILE = "sweep.csv"
NO_VAD = "no_vad"  # 對照：不做 VAD（等同 baseline）
SWEEP_COLUMNS = ("variant", "condition", "utt_id", "gender", "audio_sec", "kept_sec", "passed",
                 "S", "D", "I", "N", "hyp_norm")


def run_vad_sweep(cfg: ExperimentConfig, variants: dict[str, dict], model_name: str) -> Path:
    """對每個 VAD 設定跑調參集，逐句寫入 cfg.output_dir/sweep.csv（已完成的 variant 會跳過）。

    - 語音條件：VAD 後送模型辨識（VAD 判斷無語音時 hyp 為空、不呼叫模型），記錄 S/D/I/N 與保留秒數。
    - noise_only：只跑 VAD，記錄是否有任何片段通過（passed）；通過才可能產生幻覺，不需呼叫模型。
    - NO_VAD 對照組自動加入，只跑語音條件。

    參數:
        cfg: 調參集設定（configs/vad_tune.yaml）。
        variants: {名稱: vad 設定}（設定格式同 configs/methods/vad.yaml，可含 impl）。
        model_name: 用來量 WER 的模型。

    回傳:
        sweep.csv 路徑。
    """
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / SWEEP_FILE
    done = set(pd.read_csv(path, encoding="utf-8")["variant"]) if path.is_file() else set()
    todo = [v for v in [NO_VAD, *variants] if v not in done]
    if not todo:
        log.info("所有 variant 已完成：%s", path)
        return path
    speech, noise = split_utterances(cfg)
    conditions = [c for c in cfg.conditions if c != "noise_only"]
    normalizer = get_normalizer(cfg.normalizer)
    model = get_model(model_name)
    model.load(cfg.device, cfg.dtype, cfg.num_threads)
    try:
        for name in todo:
            vad = get_method("vad", {"vad": variants[name]}) if name != NO_VAD else None
            if vad is not None:
                vad.load(cfg.device)
            rows = []
            items = [(c, u) for c in conditions for u in speech]
            if vad is not None and "noise_only" in cfg.conditions:
                items += [("noise_only", u) for u in noise]
            for cond, u in tqdm(items, desc=name, unit="句", dynamic_ncols=True):
                wav, sr = read_audio(degraded_path(cfg, cond, u.utt_id))
                kept = vad.process(wav, sr) if vad is not None else wav
                row = {"variant": name, "condition": cond, "utt_id": u.utt_id, "gender": u.gender,
                       "audio_sec": len(wav) / sr, "kept_sec": len(kept) / sr, "passed": len(kept) > 0}
                if cond == "noise_only":
                    row.update(S=0, D=0, I=0, N=0, hyp_norm="")
                else:
                    hyp = model.transcribe(kept, sr).text if len(kept) else ""
                    a = align(normalizer(u.ref_raw), normalizer(hyp))
                    row.update(S=a.S, D=a.D, I=a.I, N=a.N, hyp_norm=normalizer(hyp))
                rows.append(row)
            if vad is not None:
                vad.unload()
            pd.DataFrame(rows, columns=list(SWEEP_COLUMNS)).to_csv(
                path, mode="a", header=not path.is_file(), index=False, encoding="utf-8", lineterminator="\n")
            log.info("完成 variant %s（%d 列）", name, len(rows))
    finally:
        model.unload()
    return path


def summarize_sweep(path: Path, hardest: str = "snrm10") -> pd.DataFrame:
    """彙總 sweep.csv：每個 variant 的各條件 WER（%）、hardest 條件的刪除數與男女 WER、純噪音通過率。

    推薦規則（事先定義，見 docs/decisions.md）：純噪音通過率不高於預設設定（variant 名稱 "default"）、
    且 clean WER 不比預設高 0.5 百分點以上的 variant 中，取 hardest 條件 WER 最低者。

    參數:
        path: sweep.csv。
        hardest: 用來選設定的最困難條件。

    回傳:
        DataFrame，index 為 variant，recommended 欄標示推薦者。
    """
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    df["errors"] = df["S"] + df["D"] + df["I"]
    sp = df[df["condition"] != "noise_only"]
    wer = sp.groupby(["variant", "condition"]).apply(
        lambda g: g["errors"].sum() / g["N"].sum() * 100, include_groups=False).unstack("condition")
    hard = sp[sp["condition"] == hardest]
    out = wer[list(dict.fromkeys(sp["condition"]))].add_prefix("wer_")
    out[f"D_{hardest}"] = hard.groupby("variant")["D"].sum()
    for g in ("female", "male"):
        hg = hard[hard["gender"] == g]
        out[f"{g}_{hardest}"] = hg.groupby("variant").apply(
            lambda x: x["errors"].sum() / x["N"].sum() * 100, include_groups=False)
    nz = df[df["condition"] == "noise_only"]
    out["noise_pass_pct"] = nz.groupby("variant")["passed"].apply(lambda s: s.astype(str).eq("True").mean() * 100)
    out = out.reindex(list(dict.fromkeys(df["variant"])))
    out["recommended"] = False
    if "default" in out.index:
        ref = out.loc["default"]
        ok = (out["noise_pass_pct"] <= ref["noise_pass_pct"]) & (out["wer_clean"] <= ref["wer_clean"] + 0.5)
        if ok.any():
            out.loc[out.loc[ok, f"wer_{hardest}"].idxmin(), "recommended"] = True
    return out
