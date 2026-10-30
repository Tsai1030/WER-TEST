"""響度正規化（pyloudnorm 於 process() 內 import）。"""

import numpy as np

from library.methods.base import AudioMethod


class LoudnormMethod(AudioMethod):
    """響度正規化到目標 LUFS（configs/methods/loudnorm.yaml 的 target_lufs，預設 −23）。

    音訊短於量測區塊（0.4 秒）或整段靜音而無法量測響度時，原樣回傳。
    """

    name = "loudnorm"

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """將整段響度正規化到目標 LUFS；超出 [-1, 1] 的部分截斷。

        例外:
            ImportError: 未安裝 pyloudnorm（uv sync --extra audio）。
        """
        if len(wav) == 0:
            return np.zeros(0, dtype=np.float32)
        import pyloudnorm

        x = wav.astype(np.float64)
        try:
            loudness = pyloudnorm.Meter(sr).integrated_loudness(x)
        except ValueError:  # 短於量測區塊
            return wav.astype(np.float32)
        if not np.isfinite(loudness):  # 整段靜音
            return wav.astype(np.float32)
        gain_db = self.config.get("target_lufs", -23.0) - loudness
        return np.clip(x * 10 ** (gain_db / 20), -1.0, 1.0).astype(np.float32)
