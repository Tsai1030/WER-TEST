"""整段語音助理：設定讀取、句子切分與各段時間計算（不需要模型、麥克風或服務）。"""

import pytest

from library.runner import load_assistant_config
from library.runner.assistant import split_sentences, turn_timings


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
