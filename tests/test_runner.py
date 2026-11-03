"""library.runner：設定讀取、矩陣展開、完整執行、中斷續跑、--no-resume（以假模型與合成資料執行）。"""

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from library.data import MANIFEST_COLUMNS
from library.export import validate_results
from library.methods import get_method
from library.models import ASRModel, Transcript
from library.runner import (
    Checkpoint,
    ExperimentConfig,
    iter_matrix,
    load_config,
    run_experiment,
    run_vad_sweep,
    split_utterances,
    summarize_sweep,
    vad_sweep,
)
from library.runner import runner as runner_module

SR = 16000
REF = "the cat sat down"  # 避免數字詞：Whisper normalizer 會把 "one two three" 合併成 "123"


class FakeModel(ASRModel):
    """回傳固定文字的假模型；crash_after 次辨識後丟出例外，模擬中斷；beams 記錄每次辨識的 beam_size。"""

    framework = "fake"
    supported_decoding = ("beam_size",)
    loads = 0
    calls = 0
    crash_after: int | None = None
    beams: ClassVar[list] = []

    def __init__(self, name: str) -> None:
        super().__init__({"hf_id": "fake/fake", "decoding": {"strategy": "greedy"}})
        self.name = name

    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        FakeModel.loads += 1
        self._model = object()

    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript:
        self._require_loaded()
        if FakeModel.crash_after is not None and FakeModel.calls >= FakeModel.crash_after:
            raise RuntimeError("模擬中斷")
        FakeModel.calls += 1
        FakeModel.beams.append(self.decoding.get("beam_size"))
        return Transcript(text="the cat sat")

    def weights_size_mb(self) -> float | None:
        return 1.0


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExperimentConfig:
    FakeModel.loads = FakeModel.calls = 0
    FakeModel.beams = []
    FakeModel.crash_after = None
    monkeypatch.setattr(runner_module, "get_model", lambda name, config=None: FakeModel(name))

    t = np.arange(2 * SR) / SR
    tone = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # Silero VAD 不會判為語音
    rows = []
    for k, (uid, gender) in enumerate((("utt001", "female"), ("utt002", "male")), start=1):
        rows.append({"utt_id": uid, "wav_path": (tmp_path / f"{uid}.wav").as_posix(), "ref_raw": REF,
                     "speaker_id": f"spk{k:02d}", "gender": gender, "age": "", "audio_sec": 2.0,
                     "source_file": f"{uid}.mp3"})
        for cond in ("clean", "snr10"):
            (tmp_path / "degraded" / cond).mkdir(parents=True, exist_ok=True)
            sf.write(tmp_path / "degraded" / cond / f"{uid}.wav", tone, SR)
    (tmp_path / "degraded" / "noise_only").mkdir(parents=True)
    sf.write(tmp_path / "degraded" / "noise_only" / "noise01.wav", tone * 0.1, SR)
    rows.append({"utt_id": "noise01", "wav_path": (tmp_path / "degraded/noise_only/noise01.wav").as_posix(),
                 "ref_raw": "", "speaker_id": "", "gender": "", "age": "", "audio_sec": 2.0, "source_file": "ENVA"})
    pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)).to_csv(tmp_path / "manifest.csv", index=False, encoding="utf-8")

    cond_file = tmp_path / "conditions.yaml"
    cond_file.write_text(yaml.safe_dump({
        "noise_source": "x", "conditions": {"clean": {}, "snr10": {"type": "noise", "snr_db": 10},
                                            "noise_only": {"type": "noise_only", "n_clips": 1, "clip_sec": 2.0}},
    }), encoding="utf-8")
    return ExperimentConfig(
        name="test", manifest=tmp_path / "manifest.csv", degraded_dir=tmp_path / "degraded",
        output_dir=tmp_path / "out", models=["granite-470m"], methods=["baseline", "vad"],
        conditions=["clean", "snr10", "noise_only"], conditions_file=cond_file, dataset="synthetic",
    )


def _results(cfg: ExperimentConfig) -> pd.DataFrame:
    return pd.read_csv(cfg.output_dir / "results.csv", encoding="utf-8", dtype=str, keep_default_na=False)


def test_iter_matrix_order(cfg: ExperimentConfig) -> None:
    combos = [(m, s, c, u.utt_id) for m, s, c, u in iter_matrix(cfg)]
    assert len(combos) == 2 * (2 * 2 + 1)
    assert combos[0] == ("granite-470m", "baseline", "clean", "utt001")
    assert combos[4] == ("granite-470m", "baseline", "noise_only", "noise01")
    assert combos[5][1] == "vad"


def test_max_utterances_is_gender_balanced() -> None:
    cfg = load_config("configs/round1.yaml")
    if not cfg.manifest.is_file():
        pytest.skip("需要 data/clean/manifest.csv（M1）")
    cfg.max_utterances = 10
    speech, _ = split_utterances(cfg)
    assert len(speech) == 10
    assert sum(u.gender == "female" for u in speech) == 5


def test_full_run_outputs_pass_validation(cfg: ExperimentConfig) -> None:
    out = run_experiment(cfg)
    assert validate_results(out) == []
    df = _results(cfg)
    assert len(df) == 10
    base = df[(df["method"] == "baseline") & (df["condition"] == "clean")].iloc[0]
    assert (base["hyp_raw"], base["S"], base["D"], base["N"], float(base["wer"])) == ("the cat sat", "0", "1", "4", 0.25)
    res = pd.read_csv(out / "resources.csv", encoding="utf-8", dtype=str, keep_default_na=False)
    assert list(res["method"]) == ["baseline", "vad"]
    import torch

    if torch.cuda.is_available():  # Jetson：偵測到 CUDA 時一律以 torch 量 GPU 記憶體
        assert (res["gpu_measure"] == "torch").all() and (res["peak_gpu_mb"] != "").all()
    else:
        assert (res["gpu_measure"] == "none").all() and (res["peak_gpu_mb"] == "").all()
    assert float(res["peak_ram_mb"].iloc[0]) > 0


def test_vad_empty_audio_skips_model(cfg: ExperimentConfig) -> None:
    run_experiment(cfg)
    df = _results(cfg)
    vad = df[df["method"] == "vad"]
    assert (vad["hyp_raw"] == "").all()  # 合成純音被 VAD 判為無語音
    assert FakeModel.calls == 5  # 只有 baseline 的 5 列呼叫模型
    assert (vad[vad["condition"] != "noise_only"]["D"] == "4").all()


def test_resume_after_crash_has_no_duplicates(cfg: ExperimentConfig) -> None:
    FakeModel.crash_after = 3
    with pytest.raises(RuntimeError, match="模擬中斷"):
        run_experiment(cfg)
    assert len(_results(cfg)) == 3
    assert len(Checkpoint(cfg.output_dir / "checkpoint.json").done) == 3  # 中斷時也會存 checkpoint

    FakeModel.crash_after = None
    out = run_experiment(cfg)
    df = _results(cfg)
    assert len(df) == 10
    assert not df.duplicated(["model", "condition", "method", "utt_id"]).any()
    assert validate_results(out) == []
    assert '"resumed": true' in (out / "meta.json").read_text(encoding="utf-8")


def test_resume_when_checkpoint_lags_results(cfg: ExperimentConfig) -> None:
    """列已寫入 results.csv 但 checkpoint 尚未存檔時中斷，續跑也不能重複。"""
    run_experiment(cfg)
    ck = cfg.output_dir / "checkpoint.json"
    ck.write_text('{"done": [], "resources": {}, "models": {}}', encoding="utf-8")
    run_experiment(cfg)
    assert len(_results(cfg)) == 10


def test_completed_model_is_not_reloaded(cfg: ExperimentConfig) -> None:
    run_experiment(cfg)
    loads, calls = FakeModel.loads, FakeModel.calls
    out = run_experiment(cfg)
    assert (FakeModel.loads, FakeModel.calls) == (loads, calls)
    assert validate_results(out) == []


def test_no_resume_backs_up_previous_outputs(cfg: ExperimentConfig) -> None:
    run_experiment(cfg)
    run_experiment(cfg, resume=False)
    assert (cfg.output_dir / "results.csv.bak").is_file()
    assert len(_results(cfg)) == 10


def test_method_params_override_reaches_method_and_meta(cfg: ExperimentConfig) -> None:
    cfg.methods = ["baseline", "loudnorm"]
    cfg.method_params = {"loudnorm": {"target_lufs": -16.0}}
    out = run_experiment(cfg)
    assert validate_results(out) == []
    meta = (out / "meta.json").read_text(encoding="utf-8")
    assert '"target_lufs": -16.0' in meta


def test_decoding_method_and_exclude(cfg: ExperimentConfig) -> None:
    cfg.models = ["granite-470m", "whisper-small"]
    cfg.methods = ["baseline", "beam", "greedy"]
    cfg.exclude = [("granite-470m", "greedy"), ("whisper-small", "beam")]
    out = run_experiment(cfg)
    assert validate_results(out) == []
    df = _results(cfg)
    assert sorted(set(map(tuple, df[["model", "method"]].drop_duplicates().to_numpy()))) == [
        ("granite-470m", "baseline"), ("granite-470m", "beam"),
        ("whisper-small", "baseline"), ("whisper-small", "greedy")]
    # 每個 (模型, 方法) 5 次辨識：baseline 不覆寫（None）、beam 5、baseline、greedy 1
    assert FakeModel.beams == [None] * 5 + [5] * 5 + [None] * 5 + [1] * 5
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["matrix"]["exclude"] == [["granite-470m", "greedy"], ["whisper-small", "beam"]]


def test_load_config_exclude_and_unsupported_decoding(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/round1.yaml").read_text(encoding="utf-8"))
    raw["methods"] = ["baseline", "beam"]
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="granite-470m 不支援 beam"):
        load_config(path)
    raw["exclude"] = [{"model": "granite-470m", "method": "beam"}, {"model": "whisper-small", "method": "beam"}]
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert load_config(path).exclude == [("granite-470m", "beam"), ("whisper-small", "beam")]
    for bad, match in (([{"model": "granite-470m", "method": "baseline"}], "baseline"),
                       ([{"model": "granite-470m", "method": "vad"}], "不在本輪"),
                       ([{"model": "granite-470m"}], "model, method")):
        raw["exclude"] = bad
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_config(path)


def test_vad_sweep_rows_resume_and_summary(cfg: ExperimentConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad_sweep, "get_model", lambda name, config=None: FakeModel(name))
    variants = {"default": {"impl": "silero"}, "webrtc_a3": {"impl": "webrtc", "aggressiveness": 3}}
    path = run_vad_sweep(cfg, variants, "granite-470m")
    df = pd.read_csv(path, encoding="utf-8")
    # no_vad：2 條件 × 2 句；每個 VAD variant 另加 1 段純噪音
    assert df.groupby("variant").size().to_dict() == {"no_vad": 4, "default": 5, "webrtc_a3": 5}
    calls = FakeModel.calls
    run_vad_sweep(cfg, variants, "granite-470m")  # 已完成的 variant 不重跑
    assert FakeModel.calls == calls and len(pd.read_csv(path, encoding="utf-8")) == 14
    s = summarize_sweep(path, hardest="snr10")
    assert s.loc["no_vad", "wer_clean"] == pytest.approx(25.0)
    assert s.loc["default", "noise_pass_pct"] == 0.0 and s["recommended"].sum() == 1


def test_load_config_round1() -> None:
    cfg = load_config("configs/round1.yaml")
    assert cfg.name == "round1" and cfg.models == ["granite-470m", "whisper-small", "qwen3-asr-0.6b"]
    assert cfg.methods == ["baseline", "vad", "denoise", "vad+denoise"]
    assert cfg.output_dir.name == "round1" and cfg.output_dir.is_absolute()
    assert cfg.nano_budget_mb == 8192 and cfg.max_utterances is None
    assert cfg.method_params == {}


def test_load_config_round2() -> None:
    cfg = load_config("configs/round2.yaml")
    assert cfg.output_dir.name == "round2"
    assert cfg.conditions == ["clean", "snr00", "snrm05", "snrm10", "lowvol", "noise_only"]
    assert {"loudnorm", "vad+loudnorm"} <= set(cfg.methods)
    assert cfg.method_params["denoise"]["params"] == {"prop_decrease": 0.5}


def test_load_config_round3() -> None:
    cfg = load_config("configs/round3.yaml")
    assert cfg.output_dir.name == "round3"
    assert cfg.methods == ["baseline", "vad", "beam", "greedy"]
    assert ("whisper-small", "beam") in cfg.exclude and ("qwen3-asr-0.6b", "greedy") in cfg.exclude
    assert cfg.method_params["vad"]["mode"] == "gate"
    assert get_method("vad", cfg.method_params).config["threshold"] == 0.3


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [("models", ["foo"], "命名規則"), ("methods", ["vad+magic"], "命名規則"),
     ("conditions", ["snr99"], "命名規則"), ("models", ["parakeet-tdt-v2"], "enabled: false"),
     ("method_params", {"magic": {}}, "命名規則")],
)
def test_load_config_rejects_bad_names(tmp_path: Path, key: str, value: list[str] | dict, match: str) -> None:
    raw = yaml.safe_load(Path("configs/round1.yaml").read_text(encoding="utf-8"))
    raw[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(path)
