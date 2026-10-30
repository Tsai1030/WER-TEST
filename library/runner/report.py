"""由 results/<round>/ 的輸出檔計算摘要（05_quick_report.py 與結果網頁共用同一套定義，見 dashboard/sections.md）。"""

import json
from pathlib import Path

import pandas as pd

from library.scoring import align, compare_per_utterance, paired_bootstrap

ALL = "all"  # 全部語音條件合併（不含 noise_only）
_KEY = ["condition", "utt_id"]


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """讀取 results.csv、resources.csv、meta.json（utf-8）。

    回傳:
        (results, resources, meta)；results 的 S/D/I/N 轉為 int，另加 errors 欄。

    例外:
        FileNotFoundError: 任一檔案不存在。
    """
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "results.csv", encoding="utf-8", dtype=str, keep_default_na=False)
    for col in ("S", "D", "I", "N"):
        df[col] = df[col].astype(int)
    for col in ("audio_sec", "latency_ms", "method_ms"):
        df[col] = df[col].astype(float)
    df["errors"] = df["S"] + df["D"] + df["I"]
    res = pd.read_csv(results_dir / "resources.csv", encoding="utf-8")
    meta = json.loads((results_dir / "meta.json").read_text(encoding="utf-8"))
    return df, res, meta


def _speech(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["condition"] != "noise_only"]


def _wer(g: pd.DataFrame) -> float:
    return g["errors"].sum() / g["N"].sum() * 100


def wer_table(df: pd.DataFrame) -> pd.DataFrame:
    """整體 WER（%，ΣS+D+I ÷ ΣN）：列為 模型 × 方法，欄為各語音條件與 all。"""
    sp = _speech(df)
    by_cond = sp.groupby(["model", "method", "condition"]).apply(_wer, include_groups=False).unstack("condition")
    by_cond[ALL] = sp.groupby(["model", "method"]).apply(_wer, include_groups=False)
    order = [c for c in dict.fromkeys(sp["condition"])] + [ALL]
    return by_cond[order]


def _verdict(lo: float, hi: float) -> str:
    if hi < 0:
        return "顯著改善"
    if lo > 0:
        return "顯著變差"
    return "無顯著差異"


def delta_table(df: pd.DataFrame, n_boot: int = 1000, seed: int = 12345) -> pd.DataFrame:
    """每個 模型 × 方法（非 baseline）× 條件（含 all）相對 baseline 的 ΔWER、95% CI、變好/變差/不變與結論。

    以「條件 × 句子」配對；all 為所有語音條件合併後配對 bootstrap。
    """
    sp = _speech(df)
    rows = []
    for model, gm in sp.groupby("model", sort=False):
        base = gm[gm["method"] == "baseline"].set_index(_KEY)
        for method in [m for m in dict.fromkeys(gm["method"]) if m != "baseline"]:
            meth = gm[gm["method"] == method].set_index(_KEY)
            for cond in [*dict.fromkeys(gm["condition"]), ALL]:
                keys = meth.index if cond == ALL else meth.index[meth.index.get_level_values(0) == cond]
                keys = keys.intersection(base.index)
                b, m = base.loc[keys], meth.loc[keys]
                lo, hi = paired_bootstrap(b["errors"].tolist(), m["errors"].tolist(), b["N"].tolist(), n_boot, seed)
                base_wer, meth_wer = _wer(b), _wer(m)
                rows.append({
                    "model": model, "method": method, "condition": cond,
                    "base_wer": base_wer, "method_wer": meth_wer, "delta_pp": meth_wer - base_wer,
                    "ci_lo": lo, "ci_hi": hi, **compare_per_utterance(b["errors"].tolist(), m["errors"].tolist()),
                    "verdict": _verdict(lo, hi),
                })
    return pd.DataFrame(rows)


def hallucination_table(df: pd.DataFrame) -> pd.DataFrame:
    """幻覺率（%）：noise_only 片段中 hyp_norm 非空的比例；列為模型，欄為方法。"""
    nz = df[df["condition"] == "noise_only"]
    rate = nz.groupby(["model", "method"])["hyp_norm"].apply(lambda s: (s != "").mean() * 100)
    return rate.unstack("method")[list(dict.fromkeys(nz["method"]))]


def normalization_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Exp0：baseline 的未正規化 WER（原始文字以空白切詞）vs 正規化 WER、假錯誤百分點與占比。"""
    sp = _speech(df)
    sp = sp[sp["method"] == "baseline"]
    rows = []
    for model, g in sp.groupby("model", sort=False):
        raw = [align(r, h) for r, h in zip(g["ref_raw"], g["hyp_raw"], strict=True)]
        raw_wer = sum(a.errors for a in raw) / sum(a.N for a in raw) * 100
        norm_wer = _wer(g)
        fake = raw_wer - norm_wer
        rows.append({"model": model, "raw_wer": raw_wer, "norm_wer": norm_wer, "false_pp": fake,
                     "false_share": fake / raw_wer * 100 if raw_wer else 0.0})
    return pd.DataFrame(rows)


def gender_table(df: pd.DataFrame) -> pd.DataFrame:
    """男女聲整體 WER（%，全部語音條件）：列為 模型 × 方法，欄為 female / male。"""
    sp = _speech(df)
    return sp.groupby(["model", "method", "gender"]).apply(_wer, include_groups=False).unstack("gender")


def cost_table(df: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """資源與延遲：峰值 RAM、平均延遲（ms，語音條件）、RTF。"""
    sp = _speech(df)
    lat = sp.groupby(["model", "method"]).agg(latency_ms=("latency_ms", "mean"),
                                              total_ms=("latency_ms", "sum"), audio=("audio_sec", "sum"))
    lat["rtf"] = lat["total_ms"] / 1000 / lat["audio"]
    out = res.set_index(["model", "method"])[["peak_ram_mb", "model_size_mb", "load_time_s"]].join(
        lat[["latency_ms", "rtf"]])
    return out


def headline(delta: pd.DataFrame) -> str:
    """一句話結論：全條件 ΔWER 最小（最負）的 模型 + 方法（定義見 dashboard/sections.md）。"""
    overall = delta[delta["condition"] == ALL]
    if overall.empty:
        return "沒有可比較的方法（只有 baseline）"
    r = overall.sort_values("delta_pp").iloc[0]
    return (f"{r['model']} + {r['method']}：WER {r['base_wer']:.2f}% → {r['method_wer']:.2f}%"
            f"（Δ {r['delta_pp']:+.2f} 百分點，95% CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]，{r['verdict']}）")


def format_report(results_dir: Path) -> str:
    """組合終端機摘要文字。"""
    df, res, meta = load_results(results_dir)
    delta = delta_table(df)
    fmt = {"float_format": lambda x: f"{x:.2f}"}
    overall = delta[delta["condition"] == ALL].set_index(["model", "method"])
    parts = [
        f"== {meta.get('round')}：{meta.get('dataset')}",
        f"   {len(df)} 列；{meta.get('cpu')}；{meta.get('precision')}；commit {meta.get('git_commit')}",
        "",
        "== 一句話結論",
        "   " + headline(delta),
        "",
        "== 整體 WER（%）",
        wer_table(df).to_string(**fmt),
        "",
        "== ΔWER（全部語音條件，百分點；負值 = 改善）",
        overall[["base_wer", "method_wer", "delta_pp", "ci_lo", "ci_hi", "improved", "worse", "same",
                 "verdict"]].to_string(**fmt),
        "",
        "== ΔWER（各條件，百分點）",
        delta[delta["condition"] != ALL].pivot_table(index=["model", "method"], columns="condition",
                                                     values="delta_pp", sort=False).to_string(**fmt),
        "",
        "== 幻覺率（%，noise_only 輸出非空的比例）",
        hallucination_table(df).to_string(**fmt),
        "",
        "== 正規化影響（Exp0，baseline）",
        normalization_impact(df).set_index("model").to_string(**fmt),
        "",
        "== 男女聲 WER（%，全部語音條件）",
        gender_table(df).to_string(**fmt),
        "",
        "== 資源與延遲（延遲為開發機 CPU 測量，僅供方法間相對比較）",
        cost_table(df, res).to_string(**fmt),
    ]
    return "\n".join(parts)
