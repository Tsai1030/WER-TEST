"""library.degrade：實測 SNR 與目標誤差、同 seed 結果相同、殘響、音量、純噪音片段。

只用合成訊號與暫存資料夾中的合成噪音源，不讀 data/。
"""

import itertools
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from library.degrade import (
    add_noise,
    add_reverb,
    apply_condition,
    build_noise_only_clips,
    change_volume,
    estimate_rt60,
    list_conditions,
    make_noise_only,
    measure_snr,
    pick_noise_env,
    prevent_clipping,
    room_impulse_response,
)

SR = 16000


def _speech(sec: float = 3.0, seed: int = 0) -> np.ndarray:
    """類語音的合成訊號：調幅的諧波，含前後靜音。"""
    t = np.arange(int(sec * SR)) / SR
    env = np.clip(np.sin(np.pi * t / sec) * 1.5, 0, 1) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    rng = np.random.default_rng(seed)
    tone = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) / k for k, f in enumerate([180, 360, 540], 1))
    return (0.2 * env * tone).astype(np.float32)


def _noise(sec: float, seed: int) -> np.ndarray:
    return (0.05 * np.random.default_rng(seed).standard_normal(int(sec * SR))).astype(np.float32)


def _exact_snr(speech: np.ndarray, mixed: np.ndarray) -> float:
    """未經整體縮放時，噪音成分就是 mixed − speech，可精確計算 SNR。"""
    s, n = speech.astype(np.float64), mixed.astype(np.float64) - speech
    return 10 * np.log10(np.mean(s**2) / np.mean(n**2))


# measure_snr 需先估計增益，語音與噪音片段的隨機相關會帶來小誤差（低 SNR 時約 0.1 dB），
# 遠小於 M2 驗收門檻 0.5 dB
MEASURE_TOL_DB = 0.2


@pytest.fixture
def cond_cfg(tmp_path: Path) -> dict:
    src = tmp_path / "noise"
    src.mkdir()
    for k, name in enumerate(["ENVA", "ENVB"]):
        sf.write(src / f"{name}.wav", _noise(40.0, k), SR)
    return {
        "noise_source": str(src),
        "noise_holdout_sec": 15.0,
        "conditions": {
            "clean": {},
            "snr10": {"type": "noise", "snr_db": 10},
            "snr00": {"type": "noise", "snr_db": 0},
            "reverb": {"type": "reverb", "rt60": 0.6, "room_dim": [5.0, 4.0, 3.0],
                       "src_mic_distance": 2.5, "geometry_seed": 42},
            "lowvol": {"type": "volume", "gain_db": -20},
            "noise_only": {"type": "noise_only", "n_clips": 5, "clip_sec": 2.0, "rms_dbfs": -32.0},
        },
    }


@pytest.mark.parametrize("snr", [20, 10, 5, 0, -5])
def test_add_noise_hits_target_snr(snr: float) -> None:
    speech, noise = _speech(), _noise(10.0, 1)
    mixed = add_noise(speech, noise, snr, seed=3)
    assert _exact_snr(speech, mixed) == pytest.approx(snr, abs=1e-3)
    assert measure_snr(speech, mixed) == pytest.approx(snr, abs=MEASURE_TOL_DB)


def test_add_noise_deterministic_and_seed_dependent() -> None:
    speech, noise = _speech(), _noise(10.0, 1)
    a = add_noise(speech, noise, 10, seed=7)
    assert np.array_equal(a, add_noise(speech, noise, 10, seed=7))
    assert not np.array_equal(a, add_noise(speech, noise, 10, seed=8))


def test_add_noise_short_noise_is_tiled() -> None:
    speech, noise = _speech(3.0), _noise(0.5, 1)
    out = add_noise(speech, noise, 5, seed=0)
    assert len(out) == len(speech)
    assert _exact_snr(speech, out) == pytest.approx(5, abs=1e-3)


def test_add_noise_rejects_empty() -> None:
    with pytest.raises(ValueError):
        add_noise(np.zeros(0, np.float32), _noise(1, 0), 10, 0)


def test_snr_survives_clip_guard_and_pcm16(tmp_path: Path) -> None:
    speech = _speech() * 4  # 放大到會削波
    mixed = prevent_clipping(add_noise(speech, _noise(10, 2), 0, seed=1))
    assert np.max(np.abs(mixed)) <= 0.99 + 1e-6
    sf.write(tmp_path / "x.wav", mixed, SR, subtype="PCM_16")
    written, _ = sf.read(tmp_path / "x.wav", dtype="float32")
    assert measure_snr(speech, written) == pytest.approx(0, abs=0.5)


def test_change_volume() -> None:
    speech = _speech()
    out = change_volume(speech, -20)
    ratio_db = 20 * np.log10(np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(speech**2)))
    assert ratio_db == pytest.approx(-20, abs=1e-3)


def test_reverb_length_level_and_determinism() -> None:
    speech = _speech()
    a = add_reverb(speech, SR, 0.6, [5.0, 4.0, 3.0], 2.5, seed=42)
    assert len(a) == len(speech)
    assert np.sqrt(np.mean(a**2)) == pytest.approx(np.sqrt(np.mean(speech**2)), rel=1e-3)
    assert np.array_equal(a, add_reverb(speech, SR, 0.6, [5.0, 4.0, 3.0], 2.5, seed=42))
    assert not np.allclose(a, speech)


def test_reverb_rt60_close_to_target() -> None:
    rir = room_impulse_response(SR, 0.6, (5.0, 4.0, 3.0), 2.5, 42)
    assert estimate_rt60(rir, SR) == pytest.approx(0.6, abs=0.15)


def test_reverb_room_too_small() -> None:
    with pytest.raises(ValueError):
        room_impulse_response(SR, 0.6, (2.0, 2.0, 3.0), 5.0, 0)


def test_make_noise_only_non_overlapping() -> None:
    noise = np.arange(10 * SR, dtype=np.float32)  # 值即位置，方便檢查重疊
    clips = make_noise_only(noise, SR, n_clips=4, clip_sec=2.0, seed=0)
    assert len(clips) == 4 and all(len(c) == 2 * SR for c in clips)
    starts = sorted(int(c[0]) for c in clips)
    assert all(b - a >= 2 * SR for a, b in itertools.pairwise(starts))
    assert all(np.array_equal(x, y) for x, y in zip(clips, make_noise_only(noise, SR, 4, 2.0, 0), strict=True))
    with pytest.raises(ValueError):
        make_noise_only(noise, SR, n_clips=6, clip_sec=2.0, seed=0)


def test_apply_condition_variants(cond_cfg: dict) -> None:
    speech = _speech()
    assert np.array_equal(apply_condition(speech, SR, "clean", cond_cfg, 0), speech)
    for cond, snr in (("snr10", 10), ("snr00", 0)):
        out = apply_condition(speech, SR, cond, cond_cfg, seed=5)
        assert _exact_snr(speech, out) == pytest.approx(snr, abs=1e-3)
    assert len(apply_condition(speech, SR, "reverb", cond_cfg, 0)) == len(speech)
    with pytest.raises(ValueError):
        apply_condition(speech, SR, "noise_only", cond_cfg, 0)
    with pytest.raises(KeyError):
        apply_condition(speech, SR, "snr99", cond_cfg, 0)


def test_same_utterance_same_noise_across_snr(cond_cfg: dict) -> None:
    """同一 seed 在不同 SNR 用同一噪音片段：兩者的噪音成分只差一個比例。"""
    speech = _speech()
    n10 = apply_condition(speech, SR, "snr10", cond_cfg, seed=9) - speech
    n00 = apply_condition(speech, SR, "snr00", cond_cfg, seed=9) - speech
    assert np.corrcoef(n10, n00)[0, 1] > 0.999
    assert pick_noise_env(cond_cfg, 9) in {"ENVA", "ENVB"}


def test_noise_only_clips_from_holdout(cond_cfg: dict) -> None:
    clips = build_noise_only_clips(cond_cfg, seed=0)
    assert [env for env, _, _ in clips] == ["ENVA", "ENVB", "ENVA", "ENVB", "ENVA"]
    for _, clip, sr in clips:
        assert sr == SR and len(clip) == 2 * SR
        assert 20 * np.log10(np.sqrt(np.mean(clip.astype(np.float64) ** 2))) == pytest.approx(-32, abs=0.01)
    # 片段必須來自最後 15 秒保留區段，不可與疊加用區段重疊
    source = sf.read(Path(cond_cfg["noise_source"]) / "ENVA.wav", dtype="float32")[0]
    holdout = source[-15 * SR :]
    raw = clips[0][1] / (clips[0][1].std() / holdout.std())  # 還原縮放前的形狀
    corr_hold = max(np.corrcoef(raw, holdout[i : i + 2 * SR])[0, 1] for i in range(0, 13 * SR + 1, 2 * SR))
    assert corr_hold > 0.99


def test_list_conditions() -> None:
    assert set(list_conditions()) == {"clean", "noise", "reverb", "volume", "noise_only"}
