"""整段語音助理：設定讀取、句子切分、各段時間計算與裝置取樣率選擇（不需要模型、麥克風或服務）。"""

from types import SimpleNamespace

import numpy as np
import pytest

from library.runner import load_assistant_config
from library.runner.assistant import compute_breakdown, device_rate, split_sentences, turn_timings


def test_config_loads_and_overrides_skip_none() -> None:
    cfg = load_assistant_config("configs/assistant.yaml", {"asr": {"device": "cpu", "dtype": None}, "tts": {"port": 9000}})
    assert cfg.asr["model"] == "granite-470m"
    assert cfg.asr["device"] == "cpu" and cfg.asr["dtype"] == "fp16"  # None 不覆寫
    assert cfg.tts["port"] == 9000 and cfg.llm["url"].startswith("http://")
    assert cfg.out_dir.name == "assistant"


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="asr.model"):
        load_assistant_config("configs/assistant.yaml", {"asr": {"model": "nope"}})
    with pytest.raises(ValueError, match="endpoint"):
        load_assistant_config("configs/assistant.yaml", {"endpoint": {"bogus": 1}})
    with pytest.raises(ValueError, match="區段"):
        load_assistant_config("configs/assistant.yaml", {"kws": {}})


def test_split_sentences_streams_complete_sentences() -> None:
    assert split_sentences("Hello there") == ([], "Hello there")
    assert split_sentences("Hello there. How") == (["Hello there."], "How")
    assert split_sentences("Yes! Sure? Ok. ") == (["Yes!", "Sure?", "Ok."], "")
    assert split_sentences("It is 3.5 degrees") == ([], "It is 3.5 degrees")  # 小數點後沒有空白，不切


def test_split_sentences_first_comma_needs_enough_words() -> None:
    assert split_sentences("Spring is warm, with rain", 3) == (["Spring is warm,"], "with rain")
    assert split_sentences("Sure, I will", 3) == ([], "Sure, I will")  # 逗號前只有 1 個字，不切
    assert split_sentences("Sure, I will remind you, okay", 3) == (["Sure, I will remind you,"], "okay")
    assert split_sentences("Spring is warm, with rain", 0) == ([], "Spring is warm, with rain")
    assert split_sentences("Yes it is. Spring is warm, so", 3) == (["Yes it is."], "Spring is warm, so")  # 有完整句子時照句子切


def test_turn_timings_relative_to_speech_end_and_stt_start() -> None:
    t = {"speech_end": 10.0, "decided": 11.0, "stt_done": 11.1, "llm_start": 11.1, "llm_first_token": 11.15,
         "llm_first_sent": 11.5, "llm_done": 11.9, "first_synth": 11.85, "all_synth": 12.2, "first_play": 11.86,
         "play_end": 15.0}
    tm = turn_timings(t)
    assert tm["wait_s"] == 1.0 and tm["stt_s"] == pytest.approx(0.1)
    assert tm["llm_ttft_s"] == pytest.approx(0.05) and tm["llm_first_sent_s"] == pytest.approx(0.4)
    assert tm["first_audio_s"] == pytest.approx(1.86) and tm["play_end_s"] == 5.0
    assert tm["stt_to_first_audio_s"] == pytest.approx(0.86) and tm["stt_to_play_end_s"] == 4.0
    assert turn_timings({k: v for k, v in t.items() if k != "llm_first_token"})["llm_ttft_s"] is None


def test_compute_breakdown_sums_components() -> None:
    c = compute_breakdown(0.0123, {"stt_s": 0.1, "llm_total_s": 0.8}, 0.6)
    assert c == {"vad_compute_s": 0.012, "compute_total_s": 1.512}
    assert compute_breakdown(0.0, {"stt_s": 0.1, "llm_total_s": None}, 0.0)["compute_total_s"] == 0.1


def test_device_rate_falls_back_to_device_default() -> None:
    class PortAudioError(Exception):
        pass

    def check(device: object, samplerate: int, channels: int) -> None:
        if samplerate not in (44100, 48000):
            raise PortAudioError("Invalid sample rate")

    sd = SimpleNamespace(PortAudioError=PortAudioError, check_input_settings=check, check_output_settings=check,
                         query_devices=lambda device, kind: {"default_samplerate": 44100.0})
    assert device_rate(sd, 24, "input", 16000) == 44100  # USB 音效卡不支援 16 kHz
    assert device_rate(sd, 24, "output", 48000) == 48000


def test_mic_stream_resampled_to_16k_keeps_length() -> None:
    import soxr

    rs = soxr.ResampleStream(44100, 16000, 1, dtype="float32")
    blocks = [rs.resample_chunk(np.zeros(4410, dtype=np.float32)) for _ in range(10)]  # 10 × 100 ms
    assert 15000 < sum(len(b) for b in blocks) <= 16000  # 只差重取樣濾波器的延遲（開頭約 20 ms）
    assert all(b.dtype == np.float32 and b.ndim == 1 for b in blocks)
