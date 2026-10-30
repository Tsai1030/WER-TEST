"""MethodChain：依序串接多個方法。"""

import numpy as np

from library.methods.base import AudioMethod


class MethodChain(AudioMethod):
    """依序套用多個方法；name 為各方法名稱以 '+' 連接（例：vad+denoise）。

    若中途某個方法回傳長度 0 的陣列，後續方法不再呼叫，直接回傳空陣列。
    """

    def __init__(self, methods: list[AudioMethod]) -> None:
        """參數:
            methods: 依處理順序排列的方法清單。

        例外:
            ValueError: methods 為空清單。
        """
        if not methods:
            raise ValueError("MethodChain 至少需要一個方法")
        super().__init__({})
        self.methods = list(methods)
        self.name = "+".join(m.name for m in self.methods)

    def load(self, device: str = "cpu") -> None:
        """依序載入所有方法。"""
        for m in self.methods:
            m.load(device)
        self.device = self.methods[0].device

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """依序處理；遇到空陣列即停止並回傳空陣列。"""
        out = wav
        for m in self.methods:
            out = m.process(out, sr)
            if len(out) == 0:
                return np.zeros(0, dtype=np.float32)
        return out

    def unload(self) -> None:
        """依序釋放所有方法。"""
        for m in self.methods:
            m.unload()

    def describe(self) -> dict:
        """回傳組合名稱與各方法的設定。"""
        return {"name": self.name, "steps": [m.describe() for m in self.methods]}
