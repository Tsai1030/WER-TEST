"""串流 TTS 服務：Kyutai Pocket TTS，透過 TCP 接收文字、邊合成邊回傳波形（協定見 library/tts/client.py）。

這是外部服務，有自己的環境（本資料夾的 pyproject.toml），不 import library。

用法（在本資料夾）:
    uv run python server.py                      # 板子：GPU（預設）
    uv run python server.py --device cpu         # 開發機：CPU
    uv run python server.py --port 8092 --voice alba
"""

import argparse
import base64
import json
import logging
import socketserver
import threading
import time

import numpy as np

log = logging.getLogger("pocket_tts_server")


def _b64(pcm: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(pcm, dtype=np.float32).tobytes()).decode("ascii")


class TTSEngine:
    """Pocket TTS 包裝：聲音狀態快取、一次只合成一句（加鎖）。"""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from pocket_tts import TTSModel

        t = time.perf_counter()
        self.model = TTSModel.load_model(language=args.language)
        if args.device == "cuda":
            self.model.to("cuda")
        torch.set_num_threads(args.threads)  # pocket_tts 在 import 時設為 1
        self.sample_rate = int(self.model.sample_rate)
        self.default_voice = args.voice
        self._states: dict = {}
        self._lock = threading.Lock()
        self.info = {"model": "pocket-tts", "language": args.language or "english", "device": args.device,
                     "voice": args.voice, "sample_rate": self.sample_rate, "stream": True}
        for text in ("Hello there, this is a warm up sentence.", "Warm up."):  # GPU 前幾次要初始化
            for _ in self.stream(text, None):
                pass
        self.info["load_s"] = round(time.perf_counter() - t, 2)

    def _state(self, voice: str | None) -> dict:
        voice = voice or self.default_voice
        if voice not in self._states:
            self._states[voice] = self.model.get_state_for_audio_prompt(voice)
        return self._states[voice]

    def stream(self, text: str, voice: str | None):
        """逐段 yield float32 波形。"""
        with self._lock:
            state = self._state(voice)
            for chunk in self.model.generate_audio_stream(state, text):
                yield chunk.detach().float().cpu().numpy().ravel()


def make_handler(engine: TTSEngine) -> type:
    """建立 socketserver 的 handler：每行一個 JSON 請求；串流請求回多行，最後一行 done。"""

    class Handler(socketserver.StreamRequestHandler):
        def _send(self, resp: dict) -> None:
            self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
            self.wfile.flush()

        def handle(self) -> None:
            for raw in self.rfile:
                try:
                    req = json.loads(raw.decode("utf-8"))
                    if req.get("cmd") == "ping":
                        self._send({"ok": True, **engine.info})
                        continue
                    text, voice = str(req["text"]), req.get("voice")
                    t0, first, n, parts = time.perf_counter(), None, 0, []
                    for pcm in engine.stream(text, voice):
                        first = first if first is not None else time.perf_counter() - t0
                        n += len(pcm)
                        if req.get("stream"):
                            self._send({"ok": True, "chunk": True, "sample_rate": engine.sample_rate,
                                        "pcm_f32_b64": _b64(pcm)})
                        else:
                            parts.append(pcm)
                    synth_s = time.perf_counter() - t0
                    resp = {"ok": True, "done": True, "sample_rate": engine.sample_rate,
                            "audio_s": round(n / engine.sample_rate, 3), "synth_s": round(synth_s, 4),
                            "first_chunk_s": round(first or synth_s, 4)}
                    if not req.get("stream"):
                        resp["pcm_f32_b64"] = _b64(np.concatenate(parts) if parts else np.zeros(0, np.float32))
                    log.info("%.3fs first, %.3fs synth, %.2fs audio: %s", resp["first_chunk_s"], synth_s,
                             resp["audio_s"], text[:60])
                except Exception as e:  # noqa: BLE001  回傳錯誤給 client，不讓服務停掉
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self._send(resp)

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="監聽位址（預設只接受本機連線）")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--language", default=None, help="Pocket TTS 語言模型（預設 english；例如 english_18m_2026-10）")
    p.add_argument("--voice", default="alba", help="預設聲音（內建名稱或 wav 檔，請求沒指定時使用）")
    p.add_argument("--threads", type=int, default=1, help="PyTorch CPU 執行緒數")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """載入模型、暖機後開始服務，直到 Ctrl+C。"""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = TTSEngine(args)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((args.host, args.port), make_handler(engine)) as srv:
        log.info("TTS_READY %s:%d %s", args.host, args.port, engine.info)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            log.info("結束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
