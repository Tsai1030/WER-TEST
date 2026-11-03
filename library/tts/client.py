"""TTS client：連到另外啟動的 TTS 服務（services/tts_server，Supertonic 3）。

協定：TCP，每行一個 JSON。請求 {"text", "voice", "lang"}；回應
{"ok": true, "sample_rate", "audio_s", "synth_s", "pcm_f32_b64"}（單聲道 float32 PCM，base64），
失敗為 {"ok": false, "error"}。{"cmd": "ping"} 回 {"ok": true, "model", "device", ...}。
"""

import base64
import json
import socket
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
    """

    wav: np.ndarray
    sample_rate: int
    synth_s: float
    audio_s: float


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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, req: dict) -> dict:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            self._f = self._sock.makefile("rw", encoding="utf-8", newline="\n")
        self._f.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._f.flush()
        line = self._f.readline()
        if not line:
            self.close()
            raise ConnectionError("TTS 服務關閉了連線")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"TTS 服務錯誤：{resp.get('error')}")
        return resp

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
        wav = np.frombuffer(base64.b64decode(r["pcm_f32_b64"]), dtype=np.float32).copy()
        return TTSResult(wav, int(r["sample_rate"]), float(r["synth_s"]), float(r["audio_s"]))

    def close(self) -> None:
        """關閉連線（之後再呼叫會重新連線）。"""
        if self._sock is not None:
            try:
                self._f.close()
                self._sock.close()
            finally:
                self._sock = self._f = None
