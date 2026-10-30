"""音量增減。"""

import numpy as np


def change_volume(speech: np.ndarray, gain_db: float) -> np.ndarray:
    """以 dB 調整音量。

    參數:
        speech: 語音陣列（一維 float32）。
        gain_db: 增益（dB），負值為降低音量。

    回傳:
        調整後的音訊；超出 [-1, 1] 的部分截斷。
    """
    raise NotImplementedError
