"""TTS client：以本機假的 TTS 服務測試協定，含串流（不需要 Supertonic / Pocket TTS）。"""

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
            elif req.get("stream"):  # 每個字一段，最後一行 done
                for _ in req["text"]:
                    pcm = np.full(10, 0.25, dtype=np.float32)
                    chunk = {"ok": True, "chunk": True, "sample_rate": 24000, "pcm_f32_b64": base64.b64encode(pcm.tobytes()).decode()}
                    self.wfile.write((json.dumps(chunk) + "\n").encode())
                n = len(req["text"]) * 10
                resp = {"ok": True, "done": True, "sample_rate": 24000, "audio_s": round(n / 24000, 3), "synth_s": 0.02,
                        "first_chunk_s": 0.005}
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


def test_synthesize_stream_yields_chunks_then_result(port) -> None:
    with TTSClient("127.0.0.1", port) as tts:
        chunks = list(tts.synthesize_stream("Hey.", voice="alba"))
        assert len(chunks) == 4 and all(len(c) == 10 and c.dtype == np.float32 for c in chunks)
        r = tts.last_result
        assert r.sample_rate == 24000 and len(r.wav) == 40 and r.first_chunk_s == 0.005
        assert len(tts.synthesize("Hi.").wav) == 30  # 串流後同一條連線仍可用
