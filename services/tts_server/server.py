"""TTS 服務：Supertonic 3，透過 TCP 接收文字、回傳波形（協定見 library/tts/client.py）。

這是外部服務，有自己的環境（本資料夾的 pyproject.toml），不 import library。

用法（在本資料夾）:
    uv run python server.py                       # 板子：GPU + 補齊 96 / 128（預設）
    uv run python server.py --device cpu --pad none   # 開發機：CPU
    uv run python server.py --port 8091 --threads 4
"""

import argparse
import base64
import json
import logging
import socketserver
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("tts_server")
MODEL_DIR = Path(__file__).resolve().parent / "models" / "supertonic-3"  # 不存在時自動下載到這裡（不進版控）
TEXT_KEYS = {"text_ids", "text_mask", "text_emb"}
LATENT_KEYS = {"noisy_latent", "latent_mask", "latent"}


def _pad(a: np.ndarray, n: int) -> np.ndarray:
    return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])])


def install_session(device: str, pad: tuple[int, int] | None) -> None:
    """在 import supertonic 之前，替換 onnxruntime 的 InferenceSession：選 CUDA、可選補齊到固定長度。

    補齊：文字與 latent 補零到固定長度，避免每次長度改變 cuDNN 卷積都要重新準備（約 0.6 秒）；
    輸出切回原長度，超過固定長度的輸入照原樣執行。補零後輸出和原版不完全相同（見 TTS Round 1）。
    """
    import onnxruntime as ort

    if device == "cuda":
        ort.preload_dlls()  # 載入 pip 安裝的 CUDA / cuDNN
        import supertonic.loader

        supertonic.loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    base = ort.InferenceSession

    class _Session(base):
        def __init__(self, path, sess_options=None, providers=None, **kw):
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})
                         if p == "CUDAExecutionProvider" else p for p in providers or []]
            super().__init__(path, sess_options=sess_options, providers=providers, **kw)

        def run(self, output_names, feed, *a, **kw):
            t = next((v.shape[-1] for k, v in feed.items() if k in TEXT_KEYS), None)
            n = next((v.shape[-1] for k, v in feed.items() if k in LATENT_KEYS), None)
            if pad is None or (t or 0) > pad[0] or (n or 0) > pad[1]:
                return super().run(output_names, feed, *a, **kw)
            feed = {k: _pad(v, pad[0]) if k in TEXT_KEYS else _pad(v, pad[1]) if k in LATENT_KEYS else v
                    for k, v in feed.items()}
            outs = []
            for o in super().run(output_names, feed, *a, **kw):
                if o.ndim >= 2 and t and o.shape[-1] == pad[0]:
                    o = o[..., :t]
                elif o.ndim >= 2 and n and o.shape[-1] == pad[1]:
                    o = o[..., :n]
                elif o.ndim >= 2 and n and "latent" in feed:  # vocoder 波形
                    o = o[..., : n * (o.shape[-1] // pad[1])]
                outs.append(o)
            return outs

    ort.InferenceSession = _Session


class TTSEngine:
    """Supertonic 3 包裝：聲音風格快取、一次只合成一句（加鎖）。"""

    def __init__(self, args: argparse.Namespace) -> None:
        from supertonic import TTS

        self.info = {"model": "supertonic-3", "device": args.device, "pad": args.pad, "steps": args.steps}
        self.steps = args.steps
        t = time.perf_counter()
        self.tts = TTS(model="supertonic-3", model_dir=args.model_dir, auto_download=True,
                       intra_op_num_threads=args.threads, inter_op_num_threads=1)
        self._styles: dict = {}
        self._lock = threading.Lock()
        self.synthesize("Warm up.", "F1", "en")  # GPU 第一次合成要初始化 CUDA / cuDNN
        self.info["load_s"] = round(time.perf_counter() - t, 2)

    def synthesize(self, text: str, voice: str, lang: str) -> dict:
        with self._lock:
            if voice not in self._styles:
                self._styles[voice] = self.tts.get_voice_style(voice_name=voice)
            t = time.perf_counter()
            wav, _ = self.tts.synthesize(text, voice_style=self._styles[voice], total_steps=self.steps, lang=lang)
            synth_s = time.perf_counter() - t
        pcm = np.ascontiguousarray(wav.ravel(), dtype=np.float32)
        return {"ok": True, "sample_rate": self.tts.sample_rate, "audio_s": round(len(pcm) / self.tts.sample_rate, 3),
                "synth_s": round(synth_s, 4), "pcm_f32_b64": base64.b64encode(pcm.tobytes()).decode("ascii")}


def make_handler(engine: TTSEngine) -> type:
    """建立 socketserver 的 handler：每行一個 JSON 請求、一行 JSON 回應，連線保持到 client 關閉。"""

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for raw in self.rfile:
                try:
                    req = json.loads(raw.decode("utf-8"))
                    if req.get("cmd") == "ping":
                        resp = {"ok": True, **engine.info}
                    else:
                        resp = engine.synthesize(str(req["text"]), str(req.get("voice", "F1")), str(req.get("lang", "en")))
                        log.info("%.3fs synth, %.2fs audio: %s", resp["synth_s"], resp["audio_s"], req["text"][:60])
                except Exception as e:  # noqa: BLE001  回傳錯誤給 client，不讓服務停掉
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                self.wfile.flush()

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="監聽位址（預設只接受本機連線）")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--pad", default="96,128", help="補齊長度「文字,latent」；none = 不補齊")
    p.add_argument("--steps", type=int, default=8, help="合成步數（Supertonic 預設 8）")
    p.add_argument("--threads", type=int, default=4, help="ONNX Runtime 執行緒數")
    p.add_argument("--model-dir", default=str(MODEL_DIR), help="模型資料夾（不存在時自動下載）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """載入模型、暖機後開始服務，直到 Ctrl+C。"""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pad = None if args.pad.lower() == "none" else tuple(int(x) for x in args.pad.split(","))
    install_session(args.device, pad)
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
