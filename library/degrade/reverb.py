"""模擬房間殘響（pyroomacoustics，於函式內 import）。"""

import numpy as np


def add_reverb(
    speech: np.ndarray,
    sr: int,
    rt60: float,
    room_dim: list[float],
    src_mic_distance: float,
    seed: int,
) -> np.ndarray:
    """以 shoebox 房間模型模擬殘響。

    參數:
        speech: 語音陣列（一維 float32）。
        sr: 取樣率。
        rt60: 殘響時間（秒）。
        room_dim: 房間尺寸 [長, 寬, 高]（公尺）。
        src_mic_distance: 聲源與麥克風距離（公尺）。
        seed: 決定聲源 / 麥克風擺放位置的亂數種子。

    回傳:
        加入殘響後的音訊，長度與 speech 相同。

    例外:
        ValueError: 房間尺寸放不下指定距離，或 rt60 無法以此房間實現。
    """
    raise NotImplementedError
