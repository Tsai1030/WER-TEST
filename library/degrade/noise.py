"""依指定 SNR 疊加噪音。"""

import numpy as np


def add_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """將噪音縮放到指定 SNR 後疊加；噪音長度不足時隨機起點循環補齊。

    noise_scaled = noise × sqrt(P_speech / (P_noise × 10^(snr/10)))，P 為整段平均功率。
    噪音片段起點以 seed 決定，確保每次產生相同結果。

    參數:
        speech: 語音陣列（一維 float32）。
        noise: 噪音陣列（一維 float32，取樣率需與 speech 相同）。
        snr_db: 目標 SNR（dB）。
        seed: 決定噪音起點的亂數種子。

    回傳:
        疊加後的音訊，長度與 speech 相同。

    例外:
        ValueError: speech 或 noise 為空陣列，或 noise 功率為 0。
    """
    raise NotImplementedError
