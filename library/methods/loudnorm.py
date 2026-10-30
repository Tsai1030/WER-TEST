"""響度 / RMS 正規化（pyloudnorm 於 load() 內 import）。"""

import numpy as np

from library.methods.base import AudioMethod


class LoudnormMethod(AudioMethod):
    """響度正規化到目標 LUFS（configs/methods/loudnorm.yaml 的 target_lufs，預設 −23）。"""

    name = "loudnorm"

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """將整段響度正規化到目標 LUFS；超出 [-1, 1] 的部分截斷。

        例外:
            ImportError: 未安裝 pyloudnorm（uv sync --extra audio）。
        """
        raise NotImplementedError
