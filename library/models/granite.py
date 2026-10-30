"""Granite Speech 5.0 470M TurboCTC（transformers，於 load() 內 import）。"""

import numpy as np

from library.models.base import ASRModel, Transcript


class GraniteModel(ASRModel):
    """Granite Speech 5.0 470M TurboCTC：Conformer + CTC，greedy 解碼。"""

    name = "granite-470m"
    framework = "transformers"

    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        """自 Hugging Face（config 的 hf_id）載入模型與 processor。"""
        raise NotImplementedError

    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript:
        """CTC greedy 解碼，回傳原始文字。"""
        raise NotImplementedError
