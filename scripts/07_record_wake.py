"""Round 4：錄製喚醒詞測試音檔（小咪小咪、音近詞、回覆聲「我在我在」）。

用法:
    uv run python scripts/07_record_wake.py --config configs/round4.yaml
    uv run python scripts/07_record_wake.py --list-devices
    uv run python scripts/07_record_wake.py --device 1

要錄的內容與次數在 configs/round4.yaml 的 record.prompts。已錄過的檔案會跳過（可分次錄完），加 --redo 重錄。
音檔存到 data/wake/（只放本機，不可上傳或公開）。錄完後執行 08_wake_eval.py --parts positive。
"""

import argparse

from library.runner import load_wake_config
from library.runner.live import run_recorder
from library.utils import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/round4.yaml", help="Round 4 設定檔")
    parser.add_argument("--device", type=int, default=None, help="輸入裝置編號（見 --list-devices；預設為系統預設麥克風）")
    parser.add_argument("--list-devices", action="store_true", help="列出音訊裝置後結束")
    parser.add_argument("--redo", action="store_true", help="重錄已存在的檔案")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """執行錄音；回傳 0 代表成功。"""
    args = parse_args(argv)
    setup_logging()
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    cfg = load_wake_config(args.config)
    run_recorder(cfg, input_device=args.device, redo=args.redo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
