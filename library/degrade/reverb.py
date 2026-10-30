"""模擬房間殘響（pyroomacoustics，於函式內 import）。"""

from functools import lru_cache

import numpy as np

_WALL_MARGIN_M = 0.5  # 聲源、麥克風與牆面的最小距離
_MIC_HEIGHT_M = 1.0  # 機器人麥克風高度
_SRC_HEIGHT_M = 1.6  # 說話者嘴巴高度
_MAX_TRIES = 1000


@lru_cache(maxsize=8)
def room_impulse_response(
    sr: int, rt60: float, room_dim: tuple[float, ...], src_mic_distance: float, seed: int
) -> np.ndarray:
    """以 shoebox 影像法（ISM）計算房間脈衝響應；相同參數只計算一次。

    牆面吸收係數由 inverse_sabine 依 rt60 與房間尺寸推得。麥克風高 1.0m、聲源高 1.6m，
    兩者三維距離為 src_mic_distance；水平位置由 seed 決定，且距牆至少 0.5m。

    參數:
        sr: 取樣率。
        rt60: 殘響時間（秒）。
        room_dim: 房間尺寸 (長, 寬, 高)（公尺）。
        src_mic_distance: 聲源與麥克風距離（公尺）。
        seed: 決定擺放位置的亂數種子。

    回傳:
        RIR（float64 一維陣列）。

    例外:
        ValueError: 房間尺寸放不下指定距離，或 rt60 無法以此房間實現。
    """
    import pyroomacoustics as pra

    length, width, height = room_dim
    dz = _SRC_HEIGHT_M - _MIC_HEIGHT_M
    if src_mic_distance <= abs(dz) or height <= _SRC_HEIGHT_M:
        raise ValueError(f"距離 {src_mic_distance}m 或房高 {height}m 不足以擺放聲源與麥克風")
    horizontal = np.sqrt(src_mic_distance**2 - dz**2)
    lo = _WALL_MARGIN_M
    rng = np.random.default_rng(seed)
    for _ in range(_MAX_TRIES):
        mic_xy = rng.uniform([lo, lo], [length - lo, width - lo])
        angle = rng.uniform(0, 2 * np.pi)
        src_xy = mic_xy + horizontal * np.array([np.cos(angle), np.sin(angle)])
        if lo <= src_xy[0] <= length - lo and lo <= src_xy[1] <= width - lo:
            break
    else:
        raise ValueError(f"房間 {room_dim} 內放不下距離 {src_mic_distance}m 的聲源與麥克風")

    e_absorption, max_order = pra.inverse_sabine(rt60, list(room_dim))
    room = pra.ShoeBox(
        list(room_dim), fs=sr, materials=pra.Material(e_absorption), max_order=max_order
    )
    room.add_source([*src_xy, _SRC_HEIGHT_M])
    room.add_microphone([*mic_xy, _MIC_HEIGHT_M])
    room.compute_rir()
    return np.asarray(room.rir[0][0], dtype=np.float64)


def estimate_rt60(rir: np.ndarray, sr: int) -> float:
    """由 RIR 估計 RT60（秒），以 30 dB 衰減外插（T30），用於驗證殘響條件。"""
    import pyroomacoustics as pra

    return float(pra.experimental.measure_rt60(rir, fs=sr, decay_db=30))


def _fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:n]


def add_reverb(
    speech: np.ndarray,
    sr: int,
    rt60: float,
    room_dim: list[float],
    src_mic_distance: float,
    seed: int,
) -> np.ndarray:
    """以 shoebox 房間模型模擬殘響。

    輸出截為與輸入同長，並將 RMS 調回與輸入相同，避免殘響條件同時改變音量（另一個變因）。

    參數:
        speech: 語音陣列（一維 float32）。
        sr: 取樣率。
        rt60: 殘響時間（秒）。
        room_dim: 房間尺寸 [長, 寬, 高]（公尺）。
        src_mic_distance: 聲源與麥克風距離（公尺）。
        seed: 決定聲源 / 麥克風擺放位置的亂數種子。

    回傳:
        加入殘響後的音訊（float32），長度與 speech 相同；未做削波處理。

    例外:
        ValueError: 房間尺寸放不下指定距離，或 rt60 無法以此房間實現。
    """
    rir = room_impulse_response(sr, rt60, tuple(room_dim), src_mic_distance, seed)
    x = speech.astype(np.float64)
    wet = _fft_convolve(x, rir)[: len(x)]
    rms_in, rms_out = np.sqrt(np.mean(x**2)), np.sqrt(np.mean(wet**2))
    if rms_out > 0:
        wet *= rms_in / rms_out
    return wet.astype(np.float32)
