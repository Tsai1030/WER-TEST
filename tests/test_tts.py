"""TTS client：以本機假的 TTS 服務測試協定（不需要 Supertonic）。"""

import base64
import json
import socketserver
import threading

import numpy as np
import pytest

from library.tts import TTSClient


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for raw in self.rfile:
            req = json.loads(raw)
            if req.get("cmd") == "ping":
                resp = {"ok": True, "model": "fake", "device": "cpu"}
            elif req["voice"] == "XX":
                resp = {"ok": False, "error": "unknown voice"}
            else:
                pcm = np.full(len(req["text"]) * 10, 0.5, dtype=np.float32)
                resp = {"ok": True, "sample_rate": 44100, "audio_s": round(len(pcm) / 44100, 3), "synth_s": 0.01,
                        "pcm_f32_b64": base64.b64encode(pcm.tobytes()).decode()}
            self.wfile.write((json.dumps(resp) + "\n").encode())


@pytest.fixture()
def port():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def test_synthesize_decodes_pcm_and_keeps_connection(port) -> None:
    with TTSClient("127.0.0.1", port) as tts:
        assert tts.ping()["model"] == "fake"
        r1 = tts.synthesize("Hello.", voice="F1", lang="en")
        r2 = tts.synthesize("Hi.", voice="M1", lang="en")  # 同一條連線送第二個請求
    assert r1.sample_rate == 44100 and r1.wav.dtype == np.float32 and len(r1.wav) == 60
    assert np.allclose(r1.wav, 0.5) and len(r2.wav) == 30


def test_error_response_raises(port) -> None:
    with TTSClient("127.0.0.1", port) as tts, pytest.raises(RuntimeError, match="unknown voice"):
        tts.synthesize("Hello.", voice="XX")
