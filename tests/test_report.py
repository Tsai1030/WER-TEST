"""library.runner.report：整體 WER、ΔWER 與結論、幻覺率、正規化影響、一句話結論。"""

import pandas as pd
import pytest

from library.runner.report import (
    ALL,
    delta_table,
    hallucination_table,
    headline,
    normalization_impact,
    wer_table,
)


def _df() -> pd.DataFrame:
    rows = []
    for k in range(20):
        for method, err in (("baseline", 2), ("vad", 1)):  # vad 每句少 1 個錯誤
            rows.append({"model": "whisper-small", "method": method, "condition": "snr10", "utt_id": f"utt{k:03d}",
                         "gender": "female" if k % 2 else "male", "S": err, "D": 0, "I": 0, "N": 10,
                         "ref_raw": "Two hundred.", "hyp_raw": "200.", "ref_norm": "200", "hyp_norm": "200"})
    for k in range(4):
        for method, hyp in (("baseline", "thank you"), ("vad", "")):
            rows.append({"model": "whisper-small", "method": method, "condition": "noise_only", "utt_id": f"noise{k:02d}",
                         "gender": "", "S": 0, "D": 0, "I": len(hyp.split()), "N": 0,
                         "ref_raw": "", "hyp_raw": hyp, "ref_norm": "", "hyp_norm": hyp})
    df = pd.DataFrame(rows)
    df["errors"] = df["S"] + df["D"] + df["I"]
    return df


def test_wer_table_is_weighted_and_excludes_noise() -> None:
    t = wer_table(_df())
    assert t.loc[("whisper-small", "baseline"), "snr10"] == pytest.approx(20.0)
    assert t.loc[("whisper-small", "vad"), ALL] == pytest.approx(10.0)
    assert "noise_only" not in t.columns


def test_delta_table_significant_improvement() -> None:
    d = delta_table(_df())
    r = d[(d["method"] == "vad") & (d["condition"] == ALL)].iloc[0]
    assert r["delta_pp"] == pytest.approx(-10.0)
    assert r["ci_hi"] < 0 and r["verdict"] == "顯著改善"
    assert (r["improved"], r["worse"], r["same"]) == (20, 0, 0)
    assert "baseline" not in set(d["method"])


def test_hallucination_rate() -> None:
    h = hallucination_table(_df())
    assert h.loc["whisper-small", "baseline"] == pytest.approx(100.0)
    assert h.loc["whisper-small", "vad"] == pytest.approx(0.0)


def test_normalization_impact_counts_false_errors() -> None:
    n = normalization_impact(_df()).iloc[0]
    # 原始文字 "Two hundred." vs "200."：S=1、D=1、N=2 → 100%；正規化後 baseline 為 20%
    assert n["raw_wer"] == pytest.approx(100.0)
    assert n["false_pp"] == pytest.approx(80.0)


def test_headline_picks_most_negative_delta() -> None:
    text = headline(delta_table(_df()))
    assert text.startswith("whisper-small + vad：WER 20.00% → 10.00%")
    assert "顯著改善" in text
