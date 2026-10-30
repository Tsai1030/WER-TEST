"""切出純噪音片段（幻覺測試用）。"""

import numpy as np


def make_noise_only(
    noise: np.ndarray, sr: int, n_clips: int, clip_sec: float, seed: int
) -> list[np.ndarray]:
    """從噪音源切出固定長度的純噪音片段。

    片段需取自和 add_noise 疊加用不同的區段，避免模型「聽過」相同噪音。

    參數:
        noise: 噪音源陣列（一維 float32）。
        sr: 取樣率。
        n_clips: 片段數。
        clip_sec: 每段秒數。
        seed: 決定片段起點的亂數種子。

    回傳:
        長度為 n_clips 的片段清單。

    例外:
        ValueError: 噪音源長度不足以切出指定片段。
    """
    raise NotImplementedError
