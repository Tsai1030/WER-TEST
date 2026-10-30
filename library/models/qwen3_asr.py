"""Qwen3-ASR-0.6B（transformers 原生版 Qwen/Qwen3-ASR-0.6B-hf，transformers ≥ 5.13，於 load() 內 import）。

不使用 qwen-asr 套件：該套件鎖定 transformers==4.57.6，與 Granite Speech 5.0（需 ≥ 5.16）衝突。
備案：若部署困難，改用 SenseVoice-Small（sherpa-onnx）。
"""

import numpy as np

from library.models.base import ASRModel, Transcript, torch_dtype
from library.utils.device import resolve_device


class Qwen3ASRModel(ASRModel):
    """Qwen3-ASR-0.6B：Audio-LLM，greedy 解碼。

    config 的 language（例：English）以 processor 預填 "language <NAME><asr_text>" 強制語言；
    設 null 則由模型自動偵測，偵測結果放在 Transcript.language。
    """

    name = "qwen3-asr-0.6b"
    framework = "transformers"

    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        """以 transformers 載入 Qwen3ASRForConditionalGeneration 與 processor。"""
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        torch.set_num_threads(num_threads)
        self.device, self.dtype = resolve_device(device), dtype
        self._processor = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.hf_id, revision=self.revision, dtype=torch_dtype(dtype)
        )
        self._model = model.to(self.device).eval()

    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript:
        """辨識並回傳原始文字；設定檔 language 為 null 時，language 欄填入自動偵測到的語言。

        例外:
            ValueError: sr 不是 16000（processor 只接受 16kHz）。
        """
        self._require_loaded()
        if sr != 16000:
            raise ValueError(f"Qwen3-ASR 需要 16kHz 音訊，實際為 {sr}")
        import torch

        language = self.config.get("language")
        decoding = self.config.get("decoding", {})
        inputs = self._processor.apply_transcription_request(audio=wav, language=language)
        inputs = inputs.to(self.device, dtype=self._model.dtype)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, do_sample=False, max_new_tokens=decoding.get("max_new_tokens", 512)
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        if language is not None:  # 強制語言，不回報偵測語言
            text = self._processor.decode(new_tokens, return_format="transcription_only")
            return Transcript(text=text.strip())
        parsed = self._processor.decode(new_tokens, return_format="parsed")
        return Transcript(text=parsed["transcription"].strip(), language=parsed["language"])
