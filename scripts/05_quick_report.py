"""在終端機印出各模型 × 方法整體 WER 摘要（不需網頁也能看結果）。

用法:
    uv run python scripts/05_quick_report.py results/round1
"""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="結果資料夾（例：results/round1）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """印出摘要（含一句話結論，需與網頁一致）；回傳 0 代表成功。"""
    parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
