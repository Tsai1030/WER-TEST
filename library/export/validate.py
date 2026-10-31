"""檢查輸出檔是否符合規格（dashboard/data_contract.md）。"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from library.export.schema import (
    CONDITION_NAMES,
    GPU_MEASURES,
    META_REQUIRED_KEYS,
    METHOD_NAMES,
    MODEL_NAMES,
    RESOURCE_COLUMNS,
    RESULT_COLUMNS,
)

_SPEAKER_ID = re.compile(r"spk\d{2,}")
_INT_COLS = ("S", "D", "I", "N")
_WER_TOL = 1e-6


def _valid_method(name: str) -> bool:
    return all(part in METHOD_NAMES for part in name.split("+"))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def _check_results(df: pd.DataFrame) -> list[str]:
    p: list[str] = []
    if tuple(df.columns) != RESULT_COLUMNS:
        return [f"results.csv 欄位不符：{list(df.columns)}"]
    for col, ok in (("model", MODEL_NAMES.__contains__), ("condition", CONDITION_NAMES.__contains__),
                    ("method", _valid_method)):
        bad = sorted({v for v in df[col] if not ok(v)})
        if bad:
            p.append(f"results.csv {col} 不符命名規則：{bad}")
    dup = df.duplicated(["model", "condition", "method", "utt_id"])
    if dup.any():
        p.append(f"results.csv 有 {int(dup.sum())} 列重複（模型×條件×方法×utt_id）")

    num = {}
    for col in (*_INT_COLS, "audio_sec", "latency_ms", "method_ms"):
        num[col] = pd.to_numeric(df[col], errors="coerce")
        if num[col].isna().any():
            p.append(f"results.csv {col} 有非數字或空值（{int(num[col].isna().sum())} 列）")
    if p:
        return p
    if (num["S"] < 0).any() or (num["D"] < 0).any() or (num["I"] < 0).any() or (num["N"] < 0).any():
        p.append("results.csv S/D/I/N 有負值")
    if (num["audio_sec"] <= 0).any():
        p.append("results.csv audio_sec 有 ≤ 0 的值")
    if (num["method_ms"] > num["latency_ms"] + 1e-6).any() or (num["latency_ms"] < 0).any():
        p.append("results.csv 有 method_ms > latency_ms 或延遲為負")

    noise = df["condition"] == "noise_only"
    wer = pd.to_numeric(df["wer"].replace("", np.nan), errors="coerce")
    errors = num["S"] + num["D"] + num["I"]
    speech_rows = ~noise & (num["N"] > 0)
    expected = errors[speech_rows] / num["N"][speech_rows]
    if (wer[speech_rows].isna()).any() or ((wer[speech_rows] - expected).abs() > _WER_TOL).any():
        p.append("results.csv 有 wer ≠ (S+D+I)/N 的語音列")
    if (~noise & (num["N"] == 0)).any():
        p.append("results.csv 有 N = 0 的語音列（ref 為空）")
    nz = df[noise]
    if len(nz):
        if (num["S"][noise] != 0).any() or (num["D"][noise] != 0).any() or (num["N"][noise] != 0).any():
            p.append("results.csv noise_only 列需 S = D = N = 0")
        if (nz["wer"] != "").any() or (nz["ref_raw"] != "").any():
            p.append("results.csv noise_only 列的 wer、ref_raw 需留空")
        hyp_words = nz["hyp_norm"].str.split().str.len().fillna(0)
        if (hyp_words != num["I"][noise]).any():
            p.append("results.csv noise_only 列需 I = hyp 詞數")
    sp = df[~noise]
    bad_spk = sorted({s for s in sp["speaker_id"] if not _SPEAKER_ID.fullmatch(s)})
    if bad_spk:
        p.append(f"results.csv speaker_id 不是匿名 ID（spkNN）：{bad_spk[:5]}")
    if not sp["gender"].isin(["female", "male"]).all():
        p.append("results.csv 語音列 gender 需為 female / male")

    for (model, cond), g in df.groupby(["model", "condition"]):
        if "baseline" not in set(g["method"]):
            p.append(f"results.csv {model} × {cond} 缺少 baseline，無法計算 ΔWER")
    return p


def _check_resources(res: pd.DataFrame, results: pd.DataFrame | None) -> list[str]:
    if tuple(res.columns) != RESOURCE_COLUMNS:
        return [f"resources.csv 欄位不符：{list(res.columns)}"]
    p = []
    bad = sorted(set(res["gpu_measure"]) - set(GPU_MEASURES))
    if bad:
        p.append(f"resources.csv gpu_measure 不合法：{bad}")
    none_with_value = (res["gpu_measure"] == "none") & (res["peak_gpu_mb"] != "")
    if none_with_value.any():
        p.append("resources.csv gpu_measure = none 時 peak_gpu_mb 需留空")
    for col in ("peak_ram_mb", "load_time_s"):
        if pd.to_numeric(res[col], errors="coerce").isna().any():
            p.append(f"resources.csv {col} 有非數字或空值")
    if res.duplicated(["model", "method"]).any():
        p.append("resources.csv 有重複的 模型×方法")
    if results is not None and tuple(results.columns) == RESULT_COLUMNS:
        need = set(map(tuple, results[["model", "method"]].drop_duplicates().to_numpy()))
        have = set(map(tuple, res[["model", "method"]].to_numpy()))
        if need - have:
            p.append(f"resources.csv 缺少：{sorted(need - have)}")
    return p


def _check_meta(meta: dict, results: pd.DataFrame | None) -> list[str]:
    p = [f"meta.json 缺少欄位：{k}" for k in META_REQUIRED_KEYS if k not in meta]
    for section, names in (("models", MODEL_NAMES), ("conditions", CONDITION_NAMES)):
        bad = sorted(k for k in meta.get(section, {}) if k not in names)
        if bad:
            p.append(f"meta.json {section} 的鍵不符命名規則：{bad}")
    bad = sorted(k for k in meta.get("methods", {}) if not _valid_method(k))
    if bad:
        p.append(f"meta.json methods 的鍵不符命名規則：{bad}")
    m = meta.get("matrix")
    if m and results is not None and tuple(results.columns) == RESULT_COLUMNS:
        n_speech_conds = sum(c != "noise_only" for c in m["conditions"])
        n_noise_conds = len(m["conditions"]) - n_speech_conds
        n_pairs = len(m["models"]) * len(m["methods"]) - len(m.get("exclude", []))
        expected = n_pairs * (n_speech_conds * m["n_speech"] + n_noise_conds * m["n_noise"])
        if len(results) != expected:
            p.append(f"results.csv 列數 {len(results)}，依 meta.matrix 應為 {expected}（是否尚未跑完？）")
    return p


def validate_results(results_dir: Path) -> list[str]:
    """檢查 results.csv、resources.csv、meta.json：欄位、命名、列數、數值範圍。

    參數:
        results_dir: 結果資料夾（例：results/round1）。

    回傳:
        問題描述清單；空清單代表通過。
    """
    results_dir = Path(results_dir)
    problems: list[str] = []
    paths = {name: results_dir / name for name in ("results.csv", "resources.csv", "meta.json")}
    missing = [n for n, path in paths.items() if not path.is_file()]
    problems += [f"缺少檔案：{results_dir / n}" for n in missing]

    results = _read_csv(paths["results.csv"]) if "results.csv" not in missing else None
    if results is not None:
        problems += _check_results(results)
    if "resources.csv" not in missing:
        problems += _check_resources(_read_csv(paths["resources.csv"]), results)
    if "meta.json" not in missing:
        try:
            meta = json.loads(paths["meta.json"].read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"meta.json 不是合法 JSON：{exc}")
        else:
            problems += _check_meta(meta, results)
    return problems
