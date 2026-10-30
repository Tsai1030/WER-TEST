"""Silero VAD：去除非語音段（silero-vad 於 load() 內 import）。"""

import numpy as np

from library.methods.base import AudioMethod


class VADMethod(AudioMethod):
    """Silero VAD：只保留語音段（前後各保留 pad_ms）。

    參數來自 configs/methods/vad.yaml：threshold、min_speech_ms、min_silence_ms、pad_ms。
    若判斷整段沒有語音，回傳長度為 0 的陣列；runner 收到空音訊時不呼叫模型（幻覺測試的關鍵）。
    """

    name = "vad"

    def load(self, device: str = "cpu") -> None:
        """載入 Silero VAD 模型。

        例外:
            ImportError: 未安裝 silero-vad（uv sync --extra audio）。
        """
        raise NotImplementedError

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """偵測語音段並串接；無語音時回傳長度 0 的陣列。

        例外:
            RuntimeError: 尚未呼叫 load()。
            ValueError: sr 不是 Silero 支援的取樣率（8000 / 16000）。
        """
        raise NotImplementedError

    def unload(self) -> None:
        """釋放 VAD 模型。"""
        raise NotImplementedError
