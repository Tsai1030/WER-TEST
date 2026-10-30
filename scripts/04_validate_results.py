"""驗證三個輸出檔（results.csv、resources.csv、meta.json）是否符合規格。

用法:
    uv run python scripts/04_validate_results.py results/round1
"""

import argparse

from library.export import validate_results
from library.utils import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="結果資料夾（例：results/round1）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """印出問題清單；回傳 0 代表通過、1 代表有問題。"""
    args = parse_args(argv)
    results_dir = resolve_path(args.results_dir)
    problems = validate_results(results_dir)
    for p in problems:
        print(f"FAIL  {p}")
    print(f"{'通過' if not problems else f'{len(problems)} 個問題'}：{results_dir}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
