"""Round 4 設定：讀取與驗證、喚醒詞 / 講完判斷參數展開、錄音清單、彙總規則。"""

from pathlib import Path

import pandas as pd
import pytest

from library.runner import endpoint_variants, kws_variants, load_wake_config, summarize_endpoint
from library.runner.live import beep, record_plan
from library.runner.wake import spotter_config
from library.utils.paths import configs_dir


def test_load_round4_config() -> None:
    cfg = load_wake_config(configs_dir() / "round4.yaml")
    assert kws_variants(cfg) == [(m, t) for m in ("kws-zh", "kws-zh-en") for t in (0.15, 0.25, 0.35)]
    names = [n for n, _ in endpoint_variants(cfg)]
    assert len(names) == 10 and names[0] == "t0.3_s300" and "t0.5_s800" not in names
    sc = spotter_config(cfg, "kws-zh-en", 0.35)
    assert sc["keywords_threshold"] == 0.35 and sc["keywords"][0]["label"] == "小咪小咪" and "thresholds" not in sc
    plan = record_plan(cfg)
    assert sum(p["kind"] == "reply" for p in plan) == 1
    assert len({p["path"] for p in plan}) == len(plan)
    assert all(p["path"].is_relative_to(cfg.wake_dir) for p in plan)


def test_load_wake_config_errors(tmp_path: Path) -> None:
    base = (configs_dir() / "round4.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    for old, new in (("thresholds: [0.15, 0.25, 0.35]", "thresholds: [1.5]"),
                     ("min_silence_ms: [300, 500, 700, 1000, 1500]", "min_silence: [300]"),
                     ("kws_model: kws-zh-en", "kws_model: nope")):
        assert old in base
        bad.write_text(base.replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError):
            load_wake_config(bad)


def test_summarize_endpoint_recommends_fastest_safe_variant(tmp_path: Path) -> None:
    rows = []
    for variant, ms, trunc, delay in (("t0.5_s300", 300, True, None), ("t0.5_s800", 800, False, 0.8),
                                      ("t0.5_s1500", 1500, False, 1.5)):
        for pause in (None, 0.3):
            cut = trunc and pause is not None
            rows.append({"variant": variant, "threshold": 0.5, "min_silence_ms": ms, "pause_sec": pause,
                         "background": "snr30", "reason": "silence", "truncated": cut,
                         "delay_sec": None if cut else delay})
    path = tmp_path / "endpoint.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    out = summarize_endpoint(path, backgrounds=("snr30",))
    assert out.loc["t0.5_s300", "scope_trunc"] == 50.0
    assert out["recommended"].sum() == 1 and out.loc["t0.5_s800", "recommended"]


def test_beep_is_short_and_quiet() -> None:
    b = beep(0.25)
    assert len(b) == 4000 and abs(b).max() <= 0.3 + 1e-6
