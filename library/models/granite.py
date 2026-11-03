"""Granite Speech 5.0 470M TurboCTC（transformers ≥ 5.16，於 load() 內 import）。"""

import numpy as np

from library.models.base import ASRModel, Transcript, torch_dtype
from library.utils.device import resolve_device


class GraniteModel(ASRModel):
    """Granite Speech 5.0 470M TurboCTC：Conformer + CTC，greedy 解碼。

    GraniteSpeech5ForCTC.generate() 即 logits argmax（CTC greedy），無其他解碼參數。
    """

    name = "granite-470m"
    framework = "transformers"

    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        """自 Hugging Face（config 的 hf_id）載入模型與 processor。"""
        import torch
        from transformers import AutoProcessor, GraniteSpeech5ForCTC

        torch.set_num_threads(num_threads)
        self.device, self.dtype = resolve_device(device), dtype
        self._processor = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision)
        model = GraniteSpeech5ForCTC.from_pretrained(
            self.hf_id, revision=self.revision, dtype=torch_dtype(dtype),
            device_map=self.device,  # 直接載入到目標裝置（Jetson 統一記憶體下避免 CPU、GPU 各存一份）
        )
        self._model = model.eval()

    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript:
        """CTC greedy 解碼，回傳原始文字。"""
        self._require_loaded()
        import torch

        inputs = self._processor(wav, sampling_rate=sr, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self._model.dtype)
        with torch.inference_mode():
            ids = self._model.generate(**inputs)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return Transcript(text=text)
