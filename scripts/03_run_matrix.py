"""執行實驗矩陣（可續跑）。

用法:
    uv run python scripts/03_run_matrix.py --config configs/round1.yaml [--no-resume] [--save-processed]
"""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round1.yaml", help="實驗設定檔")
    parser.add_argument("--no-resume", action="store_true", help="忽略 checkpoint，從頭執行")
    parser.add_argument("--save-processed", action="store_true", help="另存方法處理後的音訊（除錯用）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """執行 run_experiment()；回傳 0 代表成功。"""
    parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
