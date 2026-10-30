"""抽樣 + 轉檔 + 產生 manifest。

用法:
    uv run python scripts/01_prepare_data.py --config configs/round1.yaml
"""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round1.yaml", help="實驗設定檔")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """抽樣 50 句、轉成 16kHz mono WAV、寫出 data/clean/manifest.csv；回傳 0 代表成功。"""
    parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
