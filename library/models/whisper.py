"""Whisper small（transformers 或 faster-whisper，於 load() 內 import）。

原規劃為 large-v3-turbo；M0 實測在開發機 CPU 上約 24 s/句，改用 small（約 11.6 s/句），見 docs/decisions.md。
"""

import numpy as np

from library.models.base import ASRModel, Transcript, torch_dtype
from library.utils.device import resolve_device

_FRAMEWORKS = ("transformers", "faster-whisper")


class WhisperModel(ASRModel):
    """Whisper small：Encoder-decoder。

    framework 由設定檔決定：
    - transformers：hf_id 權重，精度為 dtype（fp32 / fp16）。
    - faster-whisper：faster_whisper_id 的 CTranslate2 權重，精度為 compute_type（例：int8）。
    解碼參數來自 configs/models/whisper-small.yaml：beam 5、temperature 0（不做 fallback）、
    condition_on_previous_text false；language 為 null 時自動偵測。faster-whisper 不啟用內建 VAD，
    VAD 一律由 methods/ 處理，避免與實驗變因混淆。
    """

    name = "whisper-small"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.framework = self.config.get("framework", "transformers")
        if self.framework not in _FRAMEWORKS:
            raise ValueError(f"{self.name} 不支援 framework：{self.framework}（可用：{_FRAMEWORKS}）")

    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        """依 config 的 framework 載入模型。"""
        self.device = resolve_device(device)
        if self.framework == "faster-whisper":
            from faster_whisper import WhisperModel as FasterWhisper

            self.dtype = self.config.get("compute_type", "int8")
            self._model = FasterWhisper(
                self.config["faster_whisper_id"],
                device="cuda" if self.device.startswith("cuda") else "cpu",
                compute_type=self.dtype,
                cpu_threads=num_threads,
            )
            return

        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        torch.set_num_threads(num_threads)
        self.dtype = dtype
        self._processor = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision)
        model = WhisperForConditionalGeneration.from_pretrained(
            self.hf_id, revision=self.revision, dtype=torch_dtype(dtype)
        )
        self._model = model.to(self.device).eval()

    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript:
        """辨識並回傳原始文字。

        例外:
            ValueError: faster-whisper 收到非 16kHz 音訊。
        """
        self._require_loaded()
        if self.framework == "faster-whisper":
            return self._transcribe_faster_whisper(wav, sr)
        import torch

        decoding = self.config.get("decoding", {})
        inputs = self._processor(wav, sampling_rate=sr, return_tensors="pt")
        features = inputs.input_features.to(self.device, dtype=self._model.dtype)
        temperature = decoding.get("temperature", 0.0)
        with torch.inference_mode():
            ids = self._model.generate(
                features,
                language=self.config.get("language"),
                task="transcribe",
                num_beams=decoding.get("beam_size", 5),
                do_sample=temperature > 0,
                **({"temperature": temperature} if temperature > 0 else {}),
            )
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return Transcript(text=text, language=self.config.get("language"))

    def _transcribe_faster_whisper(self, wav: np.ndarray, sr: int) -> Transcript:
        if sr != 16000:
            raise ValueError(f"faster-whisper 需要 16kHz 音訊，實際為 {sr}")
        decoding = self.config.get("decoding", {})
        segments, info = self._model.transcribe(
            wav.astype(np.float32),
            language=self.config.get("language"),
            task="transcribe",
            beam_size=decoding.get("beam_size", 5),
            temperature=decoding.get("temperature", 0.0),
            condition_on_previous_text=decoding.get("condition_on_previous_text", False),
            vad_filter=False,
            without_timestamps=True,
        )
        segments = list(segments)  # generator：實際解碼在此發生
        text = "".join(s.text for s in segments).strip()
        no_speech = max((s.no_speech_prob for s in segments), default=None)
        return Transcript(text=text, language=info.language, extra={"no_speech_prob": no_speech})

    def describe(self) -> dict:
        """回傳 framework、權重 ID、解碼設定與實際精度。"""
        info = super().describe()
        if self.framework == "faster-whisper":
            info["hf_id"] = self.config.get("faster_whisper_id")
            info["revision"] = None  # revision 只套用於 transformers 權重
        return info
