"""library.methods：baseline、VAD、降噪、響度正規化、解碼方法、組合與註冊表（不需下載模型）。"""

import numpy as np
import pytest

from library.data import load_manifest, read_audio
from library.export import METHOD_NAMES
from library.methods import MethodChain, get_method, list_methods
from library.methods.vad import frames_to_segments
from library.utils import resolve_path

SR = 16000
MANIFEST = resolve_path("data/clean/manifest.csv")


def _tone(sec: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _white(sec: float, amp: float, seed: int = 0) -> np.ndarray:
    return (amp * np.random.default_rng(seed).standard_normal(int(sec * SR))).astype(np.float32)


def test_registry_matches_naming_rule() -> None:
    assert sorted(list_methods()) == sorted(METHOD_NAMES)


def test_get_method_chain_and_errors() -> None:
    chain = get_method("vad+denoise")
    assert isinstance(chain, MethodChain)
    assert chain.name == "vad+denoise"
    assert [m.name for m in chain.methods] == ["vad", "denoise"]
    assert get_method("baseline").name == "baseline"
    with pytest.raises(KeyError):
        get_method("vad+magic")
    with pytest.raises(ValueError):
        get_method("vad+")


def test_get_method_reads_default_config() -> None:
    assert get_method("vad").config["threshold"] == 0.5
    assert get_method("vad", {"vad": {"threshold": 0.7}}).config["threshold"] == 0.7
    assert get_method("denoise").config["impl"] == "noisereduce"


def test_denoise_rejects_disabled_impl() -> None:
    with pytest.raises(ValueError, match="deepfilternet"):
        get_method("denoise", {"denoise": {"impl": "deepfilternet"}})


def test_baseline_identity() -> None:
    wav = _tone(1.0)
    assert np.array_equal(get_method("baseline").process(wav, SR), wav)


def test_loudnorm_hits_target() -> None:
    import pyloudnorm

    out = get_method("loudnorm").process(_tone(3.0, amp=0.01), SR)
    assert pyloudnorm.Meter(SR).integrated_loudness(out.astype(np.float64)) == pytest.approx(-23.0, abs=0.1)


def test_loudnorm_short_or_silent_unchanged() -> None:
    m = get_method("loudnorm")
    short = _tone(0.2)
    assert np.array_equal(m.process(short, SR), short)
    silent = np.zeros(SR * 2, dtype=np.float32)
    assert np.array_equal(m.process(silent, SR), silent)
    assert len(m.process(np.zeros(0, np.float32), SR)) == 0


def test_denoise_attenuates_pure_noise_keeps_length() -> None:
    """只驗證基本行為；降噪對語音是否有幫助正是 Exp2 要量的，不在單元測試中預設答案。"""
    noise = _white(3.0, 0.05)
    out = get_method("denoise").process(noise, SR)
    assert len(out) == len(noise) and out.dtype == np.float32
    drop_db = 10 * np.log10(np.mean(noise.astype(np.float64) ** 2) / np.mean(out.astype(np.float64) ** 2))
    assert drop_db > 6
    assert len(get_method("denoise").process(np.zeros(0, np.float32), SR)) == 0


def test_vad_requires_load_and_supported_sr() -> None:
    vad = get_method("vad")
    with pytest.raises(RuntimeError):
        vad.process(_tone(1.0), SR)
    vad.load("cpu")
    with pytest.raises(ValueError):
        vad.process(_tone(1.0), 44100)


def test_vad_returns_empty_for_non_speech() -> None:
    vad = get_method("vad")
    vad.load("cpu")
    assert len(vad.process(np.zeros(SR * 2, dtype=np.float32), SR)) == 0
    assert len(vad.process(_white(2.0, 0.01), SR)) == 0
    assert len(vad.process(np.zeros(0, np.float32), SR)) == 0


def test_frames_to_segments_merge_drop_pad() -> None:
    # 幀長 160（10 ms）：幀 0–4 語音、5 靜音（10 ms < min_silence 合併）、6–9 語音、20 單幀語音（短於 min_speech 捨棄）
    flags = [True] * 5 + [False] + [True] * 4 + [False] * 10 + [True] + [False] * 9
    segs = frames_to_segments(flags, 160, 30 * 160, SR, min_speech_ms=50, min_silence_ms=30, pad_ms=20)
    assert segs == [(0, 10 * 160 + 320)]


def test_webrtc_vad_and_bad_impl() -> None:
    vad = get_method("vad", {"vad": {"impl": "webrtc", "aggressiveness": 3}})
    vad.load("cpu")
    assert len(vad.process(np.zeros(SR * 2, dtype=np.float32), SR)) == 0
    assert vad.describe()["impl"] == "webrtc"
    with pytest.raises(ValueError, match="impl"):
        get_method("vad", {"vad": {"impl": "magic"}})


def test_vad_gate_keeps_whole_audio_or_nothing() -> None:
    vad = get_method("vad", {"vad": {"impl": "silero", "mode": "gate"}})
    vad.load("cpu")
    assert len(vad.process(np.zeros(SR * 2, dtype=np.float32), SR)) == 0
    vad.segments = lambda wav, sr: [(100, 200)]  # 只偵測到一小段語音：gate 仍回傳整段
    wav = _white(1.0, 0.1)
    assert np.array_equal(vad.process(wav, SR), wav)
    with pytest.raises(ValueError, match="mode"):
        get_method("vad", {"vad": {"mode": "magic"}})


def test_decoding_methods_pass_audio_and_override_beam() -> None:
    wav = _tone(0.5)
    assert get_method("beam").decoding() == {"beam_size": 5}
    assert get_method("greedy").decoding() == {"beam_size": 1}
    assert np.array_equal(get_method("beam").process(wav, SR), wav)
    assert get_method("baseline").decoding() == {} and get_method("vad").decoding() == {}
    assert get_method("vad+beam").decoding() == {"beam_size": 5}
    with pytest.raises(ValueError, match="衝突"):
        get_method("beam+greedy").decoding()


def test_chain_stops_on_empty() -> None:
    chain = get_method("vad+denoise")
    chain.load("cpu")
    assert len(chain.process(np.zeros(SR * 2, dtype=np.float32), SR)) == 0
    chain.unload()


@pytest.mark.skipif(not MANIFEST.is_file(), reason="需要 data/clean/manifest.csv（M1）")
def test_vad_keeps_speech_on_real_utterance() -> None:
    utt = next(u for u in load_manifest(MANIFEST) if u.utt_id.startswith("utt"))
    wav, sr = read_audio(utt.wav_path)
    vad = get_method("vad")
    vad.load("cpu")
    out = vad.process(wav, sr)
    assert 0.5 * len(wav) < len(out) <= len(wav)  # 去掉前後靜音，但語音大致保留
