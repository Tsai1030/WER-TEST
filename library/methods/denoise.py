"""降噪：noisereduce（原規劃 DeepFilterNet，因與新版 torchaudio 不相容而停用；套件於 process() 內 import）。"""

import numpy as np

from library.methods.base import AudioMethod

_IMPLS = ("noisereduce",)


class DenoiseMethod(AudioMethod):
    """降噪。實作由 configs/methods/denoise.yaml 的 impl 決定（目前只支援 noisereduce），
    params 直接傳給 noisereduce.reduce_noise（空 dict 即使用其預設：非穩態頻譜閘控）。
    實際使用的實作由 describe() 寫入 meta.json。
    """

    name = "denoise"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        impl = self.config.setdefault("impl", "noisereduce")
        if impl not in _IMPLS:
            raise ValueError(f"denoise 不支援 impl：{impl}（可用：{_IMPLS}；deepfilternet 已停用，見 docs/decisions.md）")

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """降噪；輸出長度與取樣率與輸入相同。

        例外:
            ImportError: 未安裝 noisereduce（uv sync --extra audio）。
        """
        if len(wav) == 0:
            return np.zeros(0, dtype=np.float32)
        import noisereduce

        out = noisereduce.reduce_noise(y=wav.astype(np.float32), sr=sr, **(self.config.get("params") or {}))
        return np.asarray(out, dtype=np.float32)
