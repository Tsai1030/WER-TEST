"""VAD：去除非語音段（Silero 或 WebRTC；套件於 load() / process() 內 import）。"""

import numpy as np

from library.methods.base import AudioMethod

_SUPPORTED_SR = (8000, 16000)
_IMPLS = ("silero", "webrtc")


def frames_to_segments(
    flags: list[bool], frame_len: int, n_samples: int, sr: int,
    min_speech_ms: float, min_silence_ms: float, pad_ms: float,
) -> list[tuple[int, int]]:
    """把逐幀語音判斷轉成語音段（樣本索引），規則與 Silero 的參數意義一致。

    依序：相鄰語音幀合併 → 間隔短於 min_silence_ms 的段合併 → 短於 min_speech_ms 的段捨棄
    → 前後各加 pad_ms（不超出音檔、重疊時合併）。

    參數:
        flags: 每幀是否為語音。
        frame_len: 每幀樣本數。
        n_samples: 音檔總樣本數。
        sr: 取樣率。
        min_speech_ms / min_silence_ms / pad_ms: 見上。

    回傳:
        [(start, end), ...]，end 不含。
    """
    segs: list[list[int]] = []
    for i, is_speech in enumerate(flags):
        if not is_speech:
            continue
        start, end = i * frame_len, min((i + 1) * frame_len, n_samples)
        if segs and start - segs[-1][1] < min_silence_ms * sr / 1000:
            segs[-1][1] = end
        else:
            segs.append([start, end])
    pad = round(pad_ms * sr / 1000)
    out: list[tuple[int, int]] = []
    for start, end in segs:
        if end - start < min_speech_ms * sr / 1000:
            continue
        start, end = max(0, start - pad), min(n_samples, end + pad)
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


class VADMethod(AudioMethod):
    """VAD：只保留語音段（前後各保留 pad_ms），各段依序串接。

    參數來自 configs/methods/vad.yaml（或該輪設定檔的 method_params）：
    impl（silero / webrtc，預設 silero）、threshold（silero）、aggressiveness（webrtc，0–3）、
    min_speech_ms、min_silence_ms、pad_ms。
    若判斷整段沒有語音，回傳長度為 0 的陣列；runner 收到空音訊時不呼叫模型（幻覺測試的關鍵）。
    VAD 模型很小，一律在 CPU 執行。
    """

    name = "vad"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        impl = self.config.setdefault("impl", "silero")
        if impl not in _IMPLS:
            raise ValueError(f"vad 不支援 impl：{impl}（可用：{_IMPLS}）")
        self._model = None

    def load(self, device: str = "cpu") -> None:
        """載入 VAD（Silero 為套件內建權重；WebRTC 為 webrtcvad-wheels），不需下載。

        例外:
            ImportError: 未安裝對應套件（uv sync --extra audio）。
        """
        self.device = "cpu"
        if self.config["impl"] == "silero":
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        else:
            import webrtcvad

            self._model = webrtcvad.Vad(int(self.config.get("aggressiveness", 2)))

    def segments(self, wav: np.ndarray, sr: int) -> list[tuple[int, int]]:
        """回傳語音段 [(start, end), ...]（樣本索引）。

        例外:
            RuntimeError: 尚未呼叫 load()。
            ValueError: sr 不是支援的取樣率（8000 / 16000）。
        """
        if self._model is None:
            raise RuntimeError("vad 尚未呼叫 load()")
        if sr not in _SUPPORTED_SR:
            raise ValueError(f"VAD 只支援 {_SUPPORTED_SR} Hz，實際為 {sr}")
        if len(wav) == 0:
            return []
        min_speech = self.config.get("min_speech_ms", 250)
        min_silence = self.config.get("min_silence_ms", 300)
        pad = self.config.get("pad_ms", 200)
        if self.config["impl"] == "webrtc":
            frame_len = sr * int(self.config.get("frame_ms", 30)) // 1000
            pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2")
            flags = [self._model.is_speech(pcm[i : i + frame_len].tobytes(), sr)
                     for i in range(0, len(pcm) - frame_len + 1, frame_len)]
            return frames_to_segments(flags, frame_len, len(wav), sr, min_speech, min_silence, pad)

        import torch
        from silero_vad import get_speech_timestamps

        stamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)),
            self._model,
            sampling_rate=sr,
            threshold=self.config.get("threshold", 0.5),
            min_speech_duration_ms=min_speech,
            min_silence_duration_ms=min_silence,
            speech_pad_ms=pad,
        )
        return [(s["start"], s["end"]) for s in stamps]

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """偵測語音段並串接；無語音時回傳長度 0 的陣列。

        例外:
            RuntimeError: 尚未呼叫 load()。
            ValueError: sr 不是支援的取樣率（8000 / 16000）。
        """
        segs = self.segments(wav, sr)
        if not segs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([wav[s:e] for s, e in segs]).astype(np.float32)

    def unload(self) -> None:
        """釋放 VAD 模型。"""
        self._model = None
