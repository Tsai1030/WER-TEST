"""切出純噪音片段（幻覺測試用）。"""

import numpy as np


def make_noise_only(
    noise: np.ndarray, sr: int, n_clips: int, clip_sec: float, seed: int
) -> list[np.ndarray]:
    """從噪音源切出固定長度、互不重疊的純噪音片段。

    片段需取自和 add_noise 疊加用不同的區段，避免模型「聽過」相同噪音；
    區段切分由呼叫端負責（registry 只把保留區段傳進來）。

    參數:
        noise: 噪音源陣列（一維 float32）。
        sr: 取樣率。
        n_clips: 片段數。
        clip_sec: 每段秒數。
        seed: 決定片段位置的亂數種子。

    回傳:
        長度為 n_clips 的片段清單，依在噪音源中的時間順序排列。

    例外:
        ValueError: 噪音源長度不足以切出指定片段。
    """
    clip_len = round(clip_sec * sr)
    n_slots = len(noise) // clip_len if clip_len > 0 else 0
    if n_clips > n_slots:
        raise ValueError(
            f"噪音長度 {len(noise) / sr:.1f}s 不足以切出 {n_clips} 段 {clip_sec}s（最多 {n_slots} 段）"
        )
    slots = np.sort(np.random.default_rng(seed).choice(n_slots, size=n_clips, replace=False))
    return [noise[i * clip_len : (i + 1) * clip_len].astype(np.float32) for i in slots]
