"""整段語音助理手動實測：用麥克風說英文 → 講完判斷 → STT → LLM → TTS → 喇叭，每一輪印出各段時間。

先另外啟動兩個服務（見 services/README.md）：
    llama-server（LLM，埠號 8090）與 services/tts_server（TTS，埠號 8091）

用法:
    uv run python scripts/10_assistant_live.py --config configs/assistant.yaml
    uv run python scripts/10_assistant_live.py --list-devices
    uv run python scripts/10_assistant_live.py --input-device 1 --output-device 2
    uv run python scripts/10_assistant_live.py --asr-device cpu --asr-dtype fp32    # 開發機
    uv run python scripts/10_assistant_live.py --min-silence-ms 700 --history-turns 2

預設值在 configs/assistant.yaml；命令列參數只覆寫本次。外放喇叭時預設回覆播完才繼續收音（--listen-during-reply 改為邊播邊收，
建議戴耳機）。每一輪的錄音、回覆音檔與時間紀錄存到 data/assistant/（只放本機，不可公開）；加 --no-save 不存。Ctrl+C 結束。
"""

import argparse

from library.runner import LiveAssistant, load_assistant_config
from library.utils import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/assistant.yaml", help="整段語音助理設定檔")
    parser.add_argument("--list-devices", action="store_true", help="列出音訊裝置後結束")
    parser.add_argument("--input-device", type=int, default=None, help="麥克風裝置編號")
    parser.add_argument("--output-device", type=int, default=None, help="喇叭裝置編號")
    parser.add_argument("--asr-device", default=None, help="STT 裝置（cuda / cpu）")
    parser.add_argument("--asr-dtype", default=None, help="STT 精度（fp32 / fp16 / bf16）")
    parser.add_argument("--min-silence-ms", type=float, default=None, help="說話後靜音多久判定講完")
    parser.add_argument("--llm-url", default=None, help="llama-server 位址（例：http://127.0.0.1:8090）")
    parser.add_argument("--tts-host", default=None, help="TTS 服務位址")
    parser.add_argument("--tts-port", type=int, default=None, help="TTS 服務埠號")
    parser.add_argument("--voice", default=None, help="TTS 聲音（F1–F5、M1–M5）")
    parser.add_argument("--history-turns", type=int, default=None, help="帶入前幾輪對話（0 = 每句獨立）")
    parser.add_argument("--listen-during-reply", action="store_true", help="回覆播放期間繼續收音（建議戴耳機）")
    parser.add_argument("--no-save", action="store_true", help="不存音檔與紀錄")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """執行即時對話直到 Ctrl+C；回傳 0 代表成功。"""
    args = parse_args(argv)
    setup_logging()
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    overrides = {
        "asr": {"device": args.asr_device, "dtype": args.asr_dtype},
        "endpoint": {"min_silence_ms": args.min_silence_ms},
        "llm": {"url": args.llm_url, "history_turns": args.history_turns},
        "tts": {"host": args.tts_host, "port": args.tts_port, "voice": args.voice},
        "live": {"input_device": args.input_device, "output_device": args.output_device,
                 "listen_during_reply": True if args.listen_during_reply else None,
                 "save": False if args.no_save else None},
    }
    LiveAssistant(load_assistant_config(args.config, overrides)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
