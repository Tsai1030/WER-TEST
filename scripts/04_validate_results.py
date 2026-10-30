"""驗證三個輸出檔（results.csv、resources.csv、meta.json）是否符合規格。

用法:
    uv run python scripts/04_validate_results.py results/round1
"""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="結果資料夾（例：results/round1）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """印出問題清單；回傳 0 代表通過、1 代表有問題。"""
    parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
