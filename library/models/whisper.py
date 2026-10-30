"""Whisper large-v3-turbo（transformers 或 faster-whisper，於 load() 內 import）。"""

import numpy as np

from library.models.base import ASRModel, Transcript


class WhisperModel(ASRModel):
    """Whisper large-v3-turbo：Encoder-decoder。

    解碼參數來自 configs/models/whisper-turbo.yaml：beam 5、temperature 0（不做 fallback）、
    condition_on_previous_text false；language 為 null 時自動偵測。
    """

    name = "whisper-turbo"
    framework = "transformers"

    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        """依 config 的 framework（transformers / faster-whisper）載入模型。"""
        raise NotImplementedError

    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript:
        """辨識並回傳原始文字；extra 中放 no_speech_prob（若可取得）。"""
        raise NotImplementedError
