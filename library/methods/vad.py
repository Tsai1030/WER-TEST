"""Silero VAD：去除非語音段（silero-vad 於 load() 內 import）。"""

import numpy as np

from library.methods.base import AudioMethod

_SUPPORTED_SR = (8000, 16000)


class VADMethod(AudioMethod):
    """Silero VAD：只保留語音段（前後各保留 pad_ms），各段依序串接。

    參數來自 configs/methods/vad.yaml：threshold、min_speech_ms、min_silence_ms、pad_ms。
    若判斷整段沒有語音，回傳長度為 0 的陣列；runner 收到空音訊時不呼叫模型（幻覺測試的關鍵）。
    VAD 模型很小，一律在 CPU 執行。
    """

    name = "vad"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._model = None

    def load(self, device: str = "cpu") -> None:
        """載入 Silero VAD 模型（套件內建權重，不需下載）。

        例外:
            ImportError: 未安裝 silero-vad（uv sync --extra audio）。
        """
        from silero_vad import load_silero_vad

        self.device = "cpu"
        self._model = load_silero_vad()

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """偵測語音段並串接；無語音時回傳長度 0 的陣列。

        例外:
            RuntimeError: 尚未呼叫 load()。
            ValueError: sr 不是 Silero 支援的取樣率（8000 / 16000）。
        """
        if self._model is None:
            raise RuntimeError("vad 尚未呼叫 load()")
        if sr not in _SUPPORTED_SR:
            raise ValueError(f"Silero VAD 只支援 {_SUPPORTED_SR} Hz，實際為 {sr}")
        if len(wav) == 0:
            return np.zeros(0, dtype=np.float32)
        import torch
        from silero_vad import get_speech_timestamps

        segments = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)),
            self._model,
            sampling_rate=sr,
            threshold=self.config.get("threshold", 0.5),
            min_speech_duration_ms=self.config.get("min_speech_ms", 250),
            min_silence_duration_ms=self.config.get("min_silence_ms", 300),
            speech_pad_ms=self.config.get("pad_ms", 200),
        )
        if not segments:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([wav[s["start"] : s["end"]] for s in segments]).astype(np.float32)

    def unload(self) -> None:
        """釋放 VAD 模型。"""
        self._model = None
