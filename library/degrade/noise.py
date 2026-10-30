"""依指定 SNR 疊加噪音，以及實測 SNR。"""

import numpy as np


def _noise_segment(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """自隨機起點取出 length 個樣本；噪音不足時循環補齊。"""
    start = int(rng.integers(0, len(noise)))
    if start + length <= len(noise):
        return noise[start : start + length]
    return noise[(start + np.arange(length)) % len(noise)]


def add_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """將噪音縮放到指定 SNR 後疊加；噪音長度不足時隨機起點循環補齊。

    noise_scaled = noise × sqrt(P_speech / (P_noise × 10^(snr/10)))，P 為整段平均功率。
    噪音片段起點以 seed 決定，確保每次產生相同結果。
    同一句在不同 SNR 條件下使用相同 seed，即可只改變 SNR、不改變噪音片段。

    參數:
        speech: 語音陣列（一維 float32）。
        noise: 噪音陣列（一維 float32，取樣率需與 speech 相同）。
        snr_db: 目標 SNR（dB）。
        seed: 決定噪音起點的亂數種子。

    回傳:
        疊加後的音訊（float32），長度與 speech 相同；未做削波處理（見 registry.prevent_clipping）。

    例外:
        ValueError: speech 或 noise 為空陣列，或 speech / noise 功率為 0。
    """
    if len(speech) == 0 or len(noise) == 0:
        raise ValueError("speech 與 noise 不可為空陣列")
    segment = _noise_segment(noise, len(speech), np.random.default_rng(seed)).astype(np.float64)
    p_speech = float(np.mean(speech.astype(np.float64) ** 2))
    p_noise = float(np.mean(segment**2))
    if p_speech == 0 or p_noise == 0:
        raise ValueError("speech 或噪音片段功率為 0，無法計算 SNR")
    scale = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    return (speech + scale * segment).astype(np.float32)


def measure_snr(clean: np.ndarray, degraded: np.ndarray) -> float:
    """由乾淨語音與加噪結果估計實際 SNR（dB）。

    加噪結果可能整體被縮放過（防削波）並經過 16-bit 量化，因此先以最小平方法估計
    degraded ≈ a·clean 的增益 a，再以 degraded − a·clean 作為噪音成分。

    參數:
        clean: 乾淨語音。
        degraded: 加噪後音訊，長度需與 clean 相同。

    回傳:
        實測 SNR（dB）。

    例外:
        ValueError: 長度不一致，或 clean 功率為 0。
    """
    if len(clean) != len(degraded):
        raise ValueError(f"長度不一致：{len(clean)} vs {len(degraded)}")
    s = clean.astype(np.float64)
    d = degraded.astype(np.float64)
    energy = float(np.dot(s, s))
    if energy == 0:
        raise ValueError("clean 功率為 0")
    a = float(np.dot(d, s)) / energy
    residual = d - a * s
    return 10 * np.log10(a * a * energy / float(np.dot(residual, residual)))
