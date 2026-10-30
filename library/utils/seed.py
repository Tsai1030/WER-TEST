"""固定亂數種子。所有亂數一律透過本模組設定，並將 seed 記錄到 meta.json。"""

import importlib.util
import random

import numpy as np


def set_seed(seed: int) -> None:
    """同時固定 random、numpy、torch（若已安裝）的亂數種子。

    參數:
        seed: 亂數種子。

    例外:
        ValueError: seed 為負數。
    """
    if seed < 0:
        raise ValueError(f"seed 不可為負數：{seed}")
    random.seed(seed)
    np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
