"""DeepFilterNet 降噪（Windows 安裝失敗時改用 noisereduce；套件於 load() 內 import）。"""

import numpy as np

from library.methods.base import AudioMethod


class DenoiseMethod(AudioMethod):
    """降噪。實作由 configs/methods/denoise.yaml 的 impl 決定（deepfilternet / noisereduce），
    實際使用的實作需寫入 meta.json。
    """

    name = "denoise"

    def load(self, device: str = "cpu") -> None:
        """載入降噪模型。

        例外:
            ImportError: 未安裝指定的降噪套件（uv sync --extra audio）。
        """
        raise NotImplementedError

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """降噪；必要時內部重取樣，輸出取樣率與輸入相同。

        例外:
            RuntimeError: 尚未呼叫 load()。
        """
        raise NotImplementedError

    def unload(self) -> None:
        """釋放降噪模型。"""
        raise NotImplementedError
