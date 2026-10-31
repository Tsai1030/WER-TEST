"""Round 4：用麥克風即時實測「小咪小咪 → 我在我在 → 說英文 → 辨識」。

用法:
    uv run python scripts/09_live_demo.py --config configs/round4.yaml
    uv run python scripts/09_live_demo.py --kws-only                 # 只測喚醒（數誤喚醒、喚醒率）
    uv run python scripts/09_live_demo.py --threshold 0.35 --kws-model kws-zh
    uv run python scripts/09_live_demo.py --min-silence-ms 1000 --no-listen-during-reply

預設值在 configs/round4.yaml 的 live 與 endpoint.default；命令列參數只覆寫本次。
外放喇叭時回覆聲會被麥克風收到，建議戴耳機，或加 --no-listen-during-reply。
每次對話的音檔與結果存到 data/wake/live/（只放本機，不可公開）；加 --no-save 不存。Ctrl+C 結束。
"""

import argparse

from library.runner import load_wake_config
from library.runner.live import LiveDemo
from library.utils import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/round4.yaml", help="Round 4 設定檔")
    parser.add_argument("--kws-model", default=None, help="喚醒詞模型（kws.models 的名稱）")
    parser.add_argument("--threshold", type=float, default=None, help="喚醒門檻（越高越不易誤喚醒）")
    parser.add_argument("--asr-model", default=None, help="辨識模型（預設 granite-470m）")
    parser.add_argument("--min-silence-ms", type=float, default=None, help="說話後靜音多久判定講完")
    parser.add_argument("--device", type=int, default=None, help="輸入裝置編號（07_record_wake.py --list-devices）")
    parser.add_argument("--kws-only", action="store_true", help="只測喚醒，不收音辨識")
    parser.add_argument("--no-listen-during-reply", action="store_true", help="回覆播放期間不收音（外放喇叭時用）")
    parser.add_argument("--single-turn", action="store_true", help="每次喚醒只收一句")
    parser.add_argument("--no-save", action="store_true", help="不存音檔與紀錄")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """執行即時實測直到 Ctrl+C；回傳 0 代表成功。"""
    args = parse_args(argv)
    setup_logging()
    cfg = load_wake_config(args.config)
    if args.min_silence_ms is not None:
        cfg.endpoint["default"] = {**cfg.endpoint.get("default", {}), "min_silence_ms": args.min_silence_ms}
    overrides = {"kws_model": args.kws_model, "threshold": args.threshold, "asr_model": args.asr_model,
                 "input_device": args.device}
    if args.no_listen_during_reply:
        overrides["listen_during_reply"] = False
    if args.single_turn:
        overrides["multi_turn"] = False
    if args.no_save:
        overrides["save"] = False
    if overrides["kws_model"] and overrides["kws_model"] not in cfg.kws["models"]:
        raise SystemExit(f"--kws-model 必須是 {list(cfg.kws['models'])} 之一")
    LiveDemo(cfg, overrides, kws_only=args.kws_only).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
