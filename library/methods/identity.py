"""baseline（不處理）。"""

import numpy as np

from library.methods.base import AudioMethod


class IdentityMethod(AudioMethod):
    """baseline：原樣回傳音訊。"""

    name = "baseline"

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """原樣回傳 wav（轉為 float32）。"""
        return wav.astype(np.float32, copy=False)
