"""產生所有劣化條件音檔（data/degraded/<condition>/<utt_id>.wav）。

用法:
    uv run python scripts/02_make_degraded.py --config configs/round1.yaml
"""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round1.yaml", help="實驗設定檔")
    parser.add_argument("--conditions", default="configs/conditions.yaml", help="劣化條件設定檔")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """產生 7 個劣化條件與純噪音片段；回傳 0 代表成功。"""
    parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
