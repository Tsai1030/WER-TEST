"""講完判斷（endpointing）：喚醒後開始收音，以串流 VAD 判斷使用者何時說完。

判斷邏輯（EndpointState）只吃逐幀語音機率，與模型分開：模擬時同一段音訊的機率算一次，
即可套用多組參數；Silero 模型於 Endpointer.load() 內 import。
"""

from dataclasses import dataclass

import numpy as np

FRAME_SAMPLES = 512  # Silero 在 16 kHz 時固定每幀 512 樣本（32 ms）
SAMPLE_RATE = 16000
REASONS = ("silence", "no_speech", "max_length", "eof")

_DEFAULTS = {
    "threshold": 0.5,  # 語音開始的機率門檻
    "neg_threshold_gap": 0.15,  # 語音中機率掉到 threshold − gap 以下才算靜音（Silero 的遲滯做法）
    "min_speech_ms": 64,  # 連續多久的語音才算開始說話（濾掉短促雜音）
    "min_silence_ms": 800,  # 說話後連續靜音多久判定講完
    "no_speech_timeout_s": 5.0,  # 開始收音後多久還沒說話就放棄
    "max_utterance_s": 15.0,  # 一句最長多久（噪音一直被判為語音時的保護）
}


@dataclass(frozen=True)
class EndpointResult:
    """一次收音的判斷結果（時間皆為相對於開始收音的秒數）。

    屬性:
        reason: silence（講完）、no_speech（逾時沒說話）、max_length（超過最長長度）、
            eof（音訊結束前還沒做出判斷，只在模擬時出現）。
        end_sec: 做出判斷的時間點；收音範圍為 [0, end_sec)。
        speech_start_sec: 判定開始說話的語音段起點（未說話為 None）。
        speech_end_sec: 最後一個語音幀的結束時間（未說話為 None）。
    """

    reason: str
    end_sec: float
    speech_start_sec: float | None
    speech_end_sec: float | None


def endpoint_params(config: dict | None = None) -> dict:
    """回傳完整參數（未給的欄位補預設值）。

    例外:
        ValueError: 有未知欄位，或數值不合理（門檻不在 0–1、時間為負）。
    """
    cfg = dict(config or {})
    unknown = set(cfg) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"endpoint 未知參數：{sorted(unknown)}（可用：{sorted(_DEFAULTS)}）")
    p = {**_DEFAULTS, **cfg}
    if not 0 < p["threshold"] <= 1:
        raise ValueError(f"threshold 必須在 (0, 1]：{p['threshold']}")
    for k in ("min_speech_ms", "min_silence_ms", "no_speech_timeout_s", "max_utterance_s", "neg_threshold_gap"):
        if p[k] < 0:
            raise ValueError(f"{k} 不可為負：{p[k]}")
    return p


class EndpointState:
    """逐幀狀態機：push(prob) 每次送入一幀的語音機率，做出判斷時回傳 EndpointResult。"""

    def __init__(self, config: dict | None = None, frame_sec: float = FRAME_SAMPLES / SAMPLE_RATE) -> None:
        self.p = endpoint_params(config)
        self.frame_sec = frame_sec
        self._neg = self.p["threshold"] - self.p["neg_threshold_gap"]
        self._min_speech = max(1, round(self.p["min_speech_ms"] / 1000 / frame_sec))
        self._i = 0  # 已送入幀數
        self._run = 0  # 目前連續語音幀數（尚未開始說話時）
        self.speech_start: float | None = None
        self.speech_end: float | None = None
        self.result: EndpointResult | None = None

    def push(self, prob: float) -> EndpointResult | None:
        """送入一幀，回傳判斷結果（尚未判斷為 None；判斷後之後的幀都回傳同一結果）。"""
        if self.result is not None:
            return self.result
        end = (self._i + 1) * self.frame_sec
        self._i += 1
        p = self.p
        if self.speech_start is None:
            self._run = self._run + 1 if prob >= p["threshold"] else 0
            if self._run >= self._min_speech:
                self.speech_start = end - self._run * self.frame_sec
                self.speech_end = end
            elif end >= p["no_speech_timeout_s"]:
                return self._done("no_speech", end)
            return None
        if prob >= self._neg:
            self.speech_end = end
        if end - self.speech_end >= p["min_silence_ms"] / 1000:
            return self._done("silence", end)
        if end - self.speech_start >= p["max_utterance_s"]:
            return self._done("max_length", end)
        return None

    def finish(self) -> EndpointResult:
        """音訊結束：回傳已做出的判斷，否則回傳 eof。"""
        return self.result or self._done("eof", self._i * self.frame_sec)

    def _done(self, reason: str, t: float) -> EndpointResult:
        self.result = EndpointResult(reason, t, self.speech_start, self.speech_end)
        return self.result


def decide(probs: np.ndarray, config: dict | None = None, frame_sec: float = FRAME_SAMPLES / SAMPLE_RATE) -> EndpointResult:
    """對一整段逐幀機率套用狀態機（模擬用）。

    參數:
        probs: 每幀語音機率（Endpointer.probs() 的輸出）。
        config: 參數，見 endpoint_params()。
        frame_sec: 每幀秒數。

    回傳:
        EndpointResult。
    """
    st = EndpointState(config, frame_sec)
    for prob in probs:
        if st.push(float(prob)) is not None:
            break
    return st.finish()


class EndpointStream:
    """即時收音用：push(chunk) 接受任意長度的音訊，內部切成 512 樣本的幀。"""

    def __init__(self, endpointer: "Endpointer", config: dict | None = None) -> None:
        self._ep = endpointer
        self.state = EndpointState(config)
        self._buf = np.zeros(0, dtype=np.float32)
        endpointer.reset()

    def push(self, chunk: np.ndarray) -> EndpointResult | None:
        """送入音訊（16 kHz float32），做出判斷時回傳 EndpointResult。"""
        self._buf = np.concatenate([self._buf, np.asarray(chunk, dtype=np.float32)])
        while len(self._buf) >= FRAME_SAMPLES and self.state.result is None:
            frame, self._buf = self._buf[:FRAME_SAMPLES], self._buf[FRAME_SAMPLES:]
            self.state.push(self._ep.frame_prob(frame))
        return self.state.result


class Endpointer:
    """以 Silero VAD 計算逐幀語音機率（只支援 16 kHz，CPU 執行）。"""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        """載入 Silero VAD（套件內建權重，不需下載）。

        例外:
            ImportError: 未安裝 silero-vad（uv sync --extra audio）。
        """
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()

    def _require(self) -> object:
        if self._model is None:
            raise RuntimeError("Endpointer 尚未呼叫 load()")
        return self._model

    def reset(self) -> None:
        """重設模型的串流狀態（每次開始收音前呼叫）。"""
        self._require().reset_states()

    def frame_prob(self, frame: np.ndarray) -> float:
        """回傳一幀（512 樣本）的語音機率；需依序送入同一條串流的幀。

        例外:
            RuntimeError: 尚未呼叫 load()。
        """
        import torch

        x = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        with torch.no_grad():
            return float(self._require()(x, SAMPLE_RATE).item())

    def probs(self, wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """從頭計算整段音訊的逐幀機率（最後不足一幀的部分捨棄）。

        例外:
            RuntimeError: 尚未呼叫 load()。
            ValueError: sr 不是 16000。
        """
        if sr != SAMPLE_RATE:
            raise ValueError(f"Endpointer 只支援 {SAMPLE_RATE} Hz，實際為 {sr}")
        self.reset()
        n = len(wav) // FRAME_SAMPLES
        return np.array([self.frame_prob(wav[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]) for i in range(n)],
                        dtype=np.float32)

    def stream(self, config: dict | None = None) -> EndpointStream:
        """建立即時收音串流（會重設模型狀態）。"""
        return EndpointStream(self, config)

    def unload(self) -> None:
        """釋放模型。"""
        self._model = None
