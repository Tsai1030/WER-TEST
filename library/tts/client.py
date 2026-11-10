"""TTS client：連到另外啟動的 TTS 服務（services/tts_server：Supertonic 3；services/pocket_tts_server：Pocket TTS，串流）。

協定：TCP，每行一個 JSON。請求 {"text", "voice", "lang"}；回應
{"ok": true, "sample_rate", "audio_s", "synth_s", "pcm_f32_b64"}（單聲道 float32 PCM，base64），
失敗為 {"ok": false, "error"}。{"cmd": "ping"} 回 {"ok": true, "model", "device", ...}；支援串流的服務另有 "stream": true。
串流請求加 "stream": true：服務每產生一段回一行 {"ok": true, "chunk": true, "sample_rate", "pcm_f32_b64"}，
最後一行 {"ok": true, "done": true, "sample_rate", "audio_s", "synth_s", "first_chunk_s"}。
"""

import base64
import json
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class TTSResult:
    """一次合成的結果。

    屬性:
        wav: 單聲道 float32 波形。
        sample_rate: 取樣率（Supertonic 為 44100）。
        synth_s: 服務端合成所花的秒數（不含傳輸）。
        audio_s: 音訊長度（秒）。
        first_chunk_s: 串流時服務端產生第一段的秒數；非串流為 None。
    """

    wav: np.ndarray
    sample_rate: int
    synth_s: float
    audio_s: float
    first_chunk_s: float | None = None


class TTSClient:
    """TTS 服務的 client；連線在第一次使用時建立並保持，可用 with 自動關閉。

    參數:
        host: 服務位址。
        port: 服務埠號。
        timeout_s: 連線與讀取逾時（秒）。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8091, timeout_s: float = 30.0) -> None:
        self.host, self.port, self.timeout_s = host, int(port), timeout_s
        self._sock: socket.socket | None = None
        self._f = None
        self.last_result: TTSResult | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _send(self, req: dict) -> None:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self._f = self._sock.makefile("rw", encoding="utf-8", newline="\n")
        self._f.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._f.flush()

    def _read(self) -> dict:
        line = self._f.readline()
        if not line:
            self.close()
            raise ConnectionError("TTS 服務關閉了連線")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"TTS 服務錯誤：{resp.get('error')}")
        return resp

    def _request(self, req: dict) -> dict:
        self._send(req)
        return self._read()

    def ping(self) -> dict:
        """確認服務可用，回傳服務資訊（模型、裝置、補齊設定等）。

        例外:
            OSError: 連不上服務。
        """
        return self._request({"cmd": "ping"})

    def synthesize(self, text: str, voice: str = "F1", lang: str = "en") -> TTSResult:
        """合成一段文字。

        參數:
            text: 要念的文字。
            voice: 聲音名稱（Supertonic：F1–F5、M1–M5）。
            lang: 語言代碼（Supertonic 3：en、ja…，不支援中文）。

        回傳:
            TTSResult。

        例外:
            OSError: 連不上服務或逾時。
            RuntimeError: 服務回傳錯誤（例如聲音名稱不存在）。
        """
        r = self._request({"text": text, "voice": voice, "lang": lang})
        wav = _pcm(r)
        self.last_result = TTSResult(wav, int(r["sample_rate"]), float(r["synth_s"]), float(r["audio_s"]),
                                     r.get("first_chunk_s"))
        return self.last_result

    def synthesize_stream(self, text: str, voice: str = "alba", lang: str = "en") -> Iterator[np.ndarray]:
        """串流合成：服務每產生一段就 yield 一段波形；讀完後整句結果放在 last_result。

        只有 ping() 回 "stream": true 的服務支援（services/pocket_tts_server）。

        參數:
            text: 要念的文字。
            voice: 聲音名稱（Pocket TTS：alba、marius…，或服務端的 wav 檔路徑）。
            lang: 語言代碼（Pocket TTS 由服務啟動時的語言模型決定，這裡不使用）。

        回傳:
            單聲道 float32 波形片段的 iterator（取樣率見 last_result.sample_rate）。

        例外:
            OSError: 連不上服務或逾時。
            RuntimeError: 服務回傳錯誤；中途出錯時這條連線會關閉，下次自動重連。
        """
        self._send({"text": text, "voice": voice, "lang": lang, "stream": True})
        parts: list[np.ndarray] = []
        try:
            while not (r := self._read()).get("done"):
                parts.append(_pcm(r))
                yield parts[-1]
        except Exception:
            self.close()  # 連線上可能還有沒讀完的片段
            raise
        wav = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        self.last_result = TTSResult(wav, int(r["sample_rate"]), float(r["synth_s"]), float(r["audio_s"]),
                                     r.get("first_chunk_s"))

    def close(self) -> None:
        """關閉連線（之後再呼叫會重新連線）。"""
        if self._sock is not None:
            try:
                self._f.close()
                self._sock.close()
            finally:
                self._sock = self._f = None


def _pcm(resp: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(resp["pcm_f32_b64"]), dtype=np.float32).copy()
