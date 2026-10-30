"""library.export：ResultWriter、write_resources、write_meta、validate_results。"""

import json
from pathlib import Path

import pandas as pd
import pytest

from library.export import (
    RESULT_COLUMNS,
    ResourceRow,
    ResultRow,
    ResultWriter,
    validate_results,
    write_meta,
    write_resources,
)
from library.runner import ExperimentConfig


def _row(method: str = "baseline", cond: str = "clean", utt: str = "utt001", **kw) -> ResultRow:
    speech = cond != "noise_only"
    base = {
        "model": "granite-470m", "condition": cond, "method": method, "utt_id": utt,
        "speaker_id": "spk01" if speech else "", "gender": "female" if speech else "", "audio_sec": 3.0,
        "ref_raw": "the cat" if speech else "", "hyp_raw": "the" if speech else "",
        "ref_norm": "the cat" if speech else "", "hyp_norm": "the" if speech else "",
        "wer": 0.5 if speech else None, "S": 0, "D": 1 if speech else 0, "I": 0,
        "N": 2 if speech else 0, "latency_ms": 10.0, "method_ms": 2.0, "detected_lang": "",
    }
    return ResultRow(**{**base, **kw})


def _write_valid(out: Path) -> None:
    with ResultWriter(out / "results.csv") as w:
        for method in ("baseline", "vad"):
            w.write(_row(method))
            w.write(_row(method, "noise_only", "noise01"))
    write_resources([ResourceRow("granite-470m", m, None, 2000.0, 900.0, 3.0, "none") for m in ("baseline", "vad")],
                    out / "resources.csv")
    cfg = ExperimentConfig(name="t", manifest=out, degraded_dir=out, output_dir=out,
                           models=["granite-470m"], methods=["baseline", "vad"], conditions=["clean", "noise_only"])
    write_meta(cfg, [], out / "meta.json", extra={
        "dataset": "synthetic", "methods": {"baseline": {}, "vad": {}}, "conditions": {"clean": {}, "noise_only": {}},
        "models": {"granite-470m": {"hf_id": "x"}},
        "matrix": {"models": cfg.models, "methods": cfg.methods, "conditions": cfg.conditions,
                   "n_speech": 1, "n_noise": 1},
    })


def _edit(out: Path, name: str, fn) -> None:
    df = pd.read_csv(out / name, encoding="utf-8", dtype=str, keep_default_na=False)
    fn(df)
    df.to_csv(out / name, index=False, encoding="utf-8", lineterminator="\n")


def test_valid_outputs_pass(tmp_path: Path) -> None:
    _write_valid(tmp_path)
    assert validate_results(tmp_path) == []
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["round"] == "t" and meta["device"] == "cpu" and meta["precision"] == "fp32"


def test_writer_header_append_and_empty_cells(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    with ResultWriter(path) as w:
        w.write(_row())
    with ResultWriter(path) as w:  # 第二次開啟為附加，不重寫標題
        w.write(_row("vad"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(RESULT_COLUMNS) and len(lines) == 3
    with ResultWriter(path) as w:
        w.write(_row(cond="noise_only", utt="noise01"))
    df = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    assert df["wer"].iloc[-1] == ""  # None → 空欄位


def test_writer_rejects_mismatched_header(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ResultWriter(tmp_path / "results.csv")


@pytest.mark.parametrize(
    ("name", "mutate", "expected"),
    [
        ("results.csv", lambda df: df.__setitem__("model", "foo"), "model 不符命名規則"),
        ("results.csv", lambda df: df.__setitem__("method", "vad+magic"), "method 不符命名規則"),
        ("results.csv", lambda df: df.loc.__setitem__((0, "wer"), "0.9"), "wer ≠"),
        ("results.csv", lambda df: df.loc.__setitem__((1, "N"), "3"), "noise_only 列需 S = D = N = 0"),
        ("results.csv", lambda df: df.loc.__setitem__((0, "method_ms"), "99"), "method_ms > latency_ms"),
        ("results.csv", lambda df: df.loc.__setitem__((0, "speaker_id"), "a1b2c3"), "匿名 ID"),
        ("results.csv", lambda df: df.drop(index=[0, 1], inplace=True), "缺少 baseline"),
        ("resources.csv", lambda df: df.loc.__setitem__((0, "peak_gpu_mb"), "12"), "peak_gpu_mb 需留空"),
        ("resources.csv", lambda df: df.drop(index=[1], inplace=True), "resources.csv 缺少"),
    ],
)
def test_validate_detects_problems(tmp_path: Path, name: str, mutate, expected: str) -> None:
    _write_valid(tmp_path)
    _edit(tmp_path, name, mutate)
    assert any(expected in p for p in validate_results(tmp_path)), validate_results(tmp_path)


def test_validate_row_count_and_missing_files(tmp_path: Path) -> None:
    _write_valid(tmp_path)
    _edit(tmp_path, "results.csv", lambda df: df.drop(index=[3], inplace=True))
    assert any("列數" in p for p in validate_results(tmp_path))
    assert any("缺少檔案" in p for p in validate_results(tmp_path / "nope"))
