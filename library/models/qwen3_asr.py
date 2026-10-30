"""Qwen3-ASR-0.6B（qwen-asr 套件、transformers 後端，於 load() 內 import）。

備案：若 Windows 部署困難，改用 SenseVoice-Small（sherpa-onnx）。
"""

import numpy as np

from library.models.base import ASRModel, Transcript


class Qwen3ASRModel(ASRModel):
    """Qwen3-ASR-0.6B：Audio-LLM，greedy 解碼，language=English。"""

    name = "qwen3-asr-0.6b"
    framework = "qwen-asr"

    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        """以 qwen-asr（transformers 後端）載入模型。

        例外:
            ImportError: 未安裝 qwen-asr（uv sync --extra models）。
        """
        raise NotImplementedError

    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript:
        """辨識並回傳原始文字；language 欄填入模型回報的語言。"""
        raise NotImplementedError
