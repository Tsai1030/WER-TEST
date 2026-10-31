"""Round 4：喚醒詞與講完判斷的自動測試，印出彙總與推薦設定。

用法:
    uv run python scripts/08_wake_eval.py --config configs/round4.yaml
    uv run python scripts/08_wake_eval.py --config configs/round4.yaml --parts endpoint
    uv run python scripts/08_wake_eval.py --config configs/round4.yaml --summary-only

parts：negative（誤喚醒，英文語音 + 噪音）、endpoint（講完判斷模擬）、positive（自錄音檔，沒有錄音時略過）。
已存在的輸出檔會跳過；要重跑某部分，先刪除 results/round4/ 下對應的 CSV。
"""

import argparse

import pandas as pd

from library.runner import (
    load_wake_config,
    run_endpoint_sim,
    run_kws_negative,
    run_kws_positive,
    summarize_endpoint,
    summarize_kws,
)
from library.utils import set_seed, setup_logging

PARTS = ("negative", "endpoint", "positive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/round4.yaml", help="Round 4 設定檔")
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS), help="要執行的部分")
    parser.add_argument("--summary-only", action="store_true", help="不執行，只彙總既有輸出")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """執行測試並印出彙總；回傳 0 代表成功。"""
    args = parse_args(argv)
    setup_logging()
    cfg = load_wake_config(args.config)
    set_seed(cfg.seed)
    if not args.summary_only:
        if "negative" in args.parts:
            run_kws_negative(cfg)
        if "endpoint" in args.parts:
            run_endpoint_sim(cfg)
        if "positive" in args.parts:
            run_kws_positive(cfg)
    fmt = lambda x: f"{x:.2f}"
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        for name, table in summarize_kws(cfg.output_dir).items():
            print(f"\n== 喚醒詞：{'誤喚醒（n_false 次、fa_per_hour 每小時次數）' if name == 'negative' else '正確喚醒率（%）與音近詞'}")
            print(table.to_string(float_format=fmt))
        ep_path = cfg.output_dir / "endpoint.csv"
        if ep_path.is_file():
            print("\n== 講完判斷（截斷率 / 失控率 / 漏收率為 %，等待時間為秒）")
            print(summarize_endpoint(ep_path).to_string(float_format=fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
