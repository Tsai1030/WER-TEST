"""講完判斷：狀態機（以假的逐幀機率測試，不需模型）與參數檢查；Silero 串流與離線結果一致。"""

import numpy as np
import pytest

from library.endpoint import FRAME_SAMPLES, Endpointer, EndpointState, decide, endpoint_params

F = FRAME_SAMPLES / 16000  # 0.032 s


def _probs(*runs: tuple[float, float]) -> np.ndarray:
    """[(機率, 秒數), ...] → 逐幀機率。"""
    return np.concatenate([np.full(round(sec / F), p, dtype=np.float32) for p, sec in runs])


def test_silence_after_speech_ends_utterance() -> None:
    r = decide(_probs((0.0, 0.5), (0.9, 2.0), (0.0, 3.0)), {"min_silence_ms": 800})
    assert r.reason == "silence"
    assert r.speech_start_sec == pytest.approx(0.5, abs=F)
    assert r.speech_end_sec == pytest.approx(2.5, abs=F)
    assert r.end_sec - r.speech_end_sec == pytest.approx(0.8, abs=F)


def test_short_pause_does_not_end_but_long_pause_does() -> None:
    probs = _probs((0.9, 1.0), (0.0, 0.5), (0.9, 1.0), (0.0, 2.0))
    assert decide(probs, {"min_silence_ms": 800}).speech_end_sec == pytest.approx(2.5, abs=F)
    assert decide(probs, {"min_silence_ms": 300}).end_sec == pytest.approx(1.3, abs=F)  # 句中停頓被截斷


def test_hysteresis_keeps_speech_between_thresholds() -> None:
    probs = _probs((0.9, 1.0), (0.4, 1.0), (0.0, 2.0))  # 0.4 高於 0.5 - 0.15，仍算語音
    assert decide(probs, {"min_silence_ms": 500}).speech_end_sec == pytest.approx(2.0, abs=F)


def test_no_speech_max_length_eof_and_short_blips() -> None:
    assert decide(_probs((0.0, 6.0)), {"no_speech_timeout_s": 5.0}).reason == "no_speech"
    assert decide(_probs((0.9, 0.032), (0.0, 6.0)), {"min_speech_ms": 64}).reason == "no_speech"  # 單幀雜音不算
    assert decide(_probs((0.9, 20.0)), {"max_utterance_s": 15.0}).reason == "max_length"
    assert decide(_probs((0.9, 3.0)), {"max_utterance_s": 15.0}).reason == "eof"


def test_state_returns_same_result_after_decision() -> None:
    st = EndpointState({"min_silence_ms": 100})
    results = [st.push(p) for p in _probs((0.9, 0.5), (0.0, 1.0))]
    decided = [r for r in results if r is not None]
    assert decided and all(r is decided[0] for r in decided)


def test_endpoint_params_validation() -> None:
    assert endpoint_params()["min_silence_ms"] == 800
    with pytest.raises(ValueError):
        endpoint_params({"min_silence": 800})
    with pytest.raises(ValueError):
        endpoint_params({"threshold": 1.5})
    with pytest.raises(ValueError):
        endpoint_params({"min_silence_ms": -1})


def test_stream_matches_offline_probs() -> None:
    ep = Endpointer()
    with pytest.raises(RuntimeError):
        ep.probs(np.zeros(16000, dtype=np.float32))
    ep.load()
    try:
        rng = np.random.default_rng(0)
        wav = (0.01 * rng.standard_normal(16000 * 2)).astype(np.float32)
        offline = decide(ep.probs(wav), {"no_speech_timeout_s": 1.0})
        stream = ep.stream({"no_speech_timeout_s": 1.0})
        online = None
        for i in range(0, len(wav), 1600):
            online = stream.push(wav[i : i + 1600]) or online
        assert offline == online
        with pytest.raises(ValueError):
            ep.probs(wav, 8000)
    finally:
        ep.unload()
