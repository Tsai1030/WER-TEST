"""條件名稱 → 處理函式的對應表。"""

import numpy as np


def apply_condition(speech: np.ndarray, sr: int, condition: str, cfg: dict, seed: int) -> np.ndarray:
    """依條件名稱（clean / snr20 / reverb …）套用對應處理。

    參數:
        speech: 語音陣列（一維 float32）。
        sr: 取樣率。
        condition: 條件名稱（見 plan.md 6.5）。
        cfg: configs/conditions.yaml 讀入後的 dict（含 noise_source 與 conditions）。
        seed: 亂數種子。

    回傳:
        劣化後的音訊。clean 原樣回傳。

    例外:
        KeyError: 條件名稱不在 cfg 中。
        ValueError: noise_only 不可用本函式產生（請用 make_noise_only）。
    """
    raise NotImplementedError


def list_conditions() -> list[str]:
    """回傳所有已註冊的條件名稱。"""
    raise NotImplementedError
