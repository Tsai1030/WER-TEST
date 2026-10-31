"""Round 3：在調參集上掃描 VAD 參數 / 實作，印出彙總與推薦設定。

用法:
    uv run python scripts/01_prepare_data.py --config configs/vad_tune.yaml
    uv run python scripts/02_make_degraded.py --config configs/vad_tune.yaml --conditions configs/conditions_tune.yaml
    uv run python scripts/06_vad_sweep.py --config configs/vad_tune.yaml

已完成的 variant 會跳過（可中斷後重跑）；只印彙總時加 --summary-only。
"""

import argparse

import pandas as pd

from library.runner import load_config, run_vad_sweep, summarize_sweep
from library.utils import load_yaml, resolve_path, set_seed, setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vad_tune.yaml", help="調參集設定檔")
    parser.add_argument("--summary-only", action="store_true", help="不執行，只彙總既有 sweep.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """執行掃描並印出彙總；回傳 0 代表成功。"""
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    raw = load_yaml(resolve_path(args.config))
    set_seed(cfg.seed)
    sweep = raw["sweep"]
    path = cfg.output_dir / "sweep.csv"
    if not args.summary_only:
        path = run_vad_sweep(cfg, sweep["variants"], sweep["model"])
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summarize_sweep(path, sweep.get("hardest", "snrm10")).to_string(float_format=lambda x: f"{x:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
