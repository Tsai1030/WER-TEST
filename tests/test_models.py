"""library.models 註冊表與介面（不載入模型權重），以及 runner.bench 的矩陣規模計算。"""

import numpy as np
import pytest

from library.export import MODEL_NAMES
from library.models import get_model, list_models
from library.runner import inferences_per_model
from library.utils import resolve_device


def test_registry_matches_naming_rule() -> None:
    assert sorted(list_models()) == sorted(MODEL_NAMES)


def test_unknown_model() -> None:
    with pytest.raises(KeyError):
        get_model("no-such-model")


def test_parakeet_disabled() -> None:
    with pytest.raises(RuntimeError, match="enabled: false"):
        get_model("parakeet-tdt-v2")


@pytest.mark.parametrize("name", ["granite-470m", "whisper-small", "qwen3-asr-0.6b"])
def test_enabled_models_construct_without_loading(name: str) -> None:
    model = get_model(name)
    assert model.name == name
    info = model.describe()
    assert info["hf_id"] and info["framework"] == "transformers"
    with pytest.raises(RuntimeError, match="load"):
        model.transcribe(np.zeros(16000, dtype=np.float32), 16000)


def test_resolve_device_falls_back_to_cpu() -> None:
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("cuda") == expected
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_inferences_per_model_round1() -> None:
    conditions = ["clean", "snr20", "snr10", "snr05", "snr00", "reverb", "lowvol", "noise_only"]
    assert inferences_per_model(4, conditions, n_speech=50, n_noise=10) == 1440
