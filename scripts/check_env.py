"""檢查 Python、套件版本、可用裝置（CPU / CUDA）、模型能否載入，並實測每句秒數。

用法:
    uv run python scripts/check_env.py --models granite-470m whisper-turbo --bench 5
"""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round1.yaml", help="實驗設定檔")
    parser.add_argument("--models", nargs="*", default=None, help="要檢查的模型；預設為設定檔中的全部")
    parser.add_argument("--bench", type=int, default=0, help="每個模型實測幾句 clean 音檔（0 = 不實測）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """執行環境檢查；回傳 0 代表全部通過。"""
    parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
