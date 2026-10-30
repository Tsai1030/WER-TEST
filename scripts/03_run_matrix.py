"""執行實驗矩陣（可續跑）。

用法:
    uv run python scripts/03_run_matrix.py --config configs/round1.yaml [--no-resume] [--save-processed]

中斷（Ctrl+C 或關機）後用同一指令重跑即從中斷處繼續；--no-resume 會把舊輸出改名為 .bak 後從頭開始。
"""

import argparse

from library.export import validate_results
from library.runner import load_config, run_experiment
from library.utils import get_logger, set_seed, setup_logging

log = get_logger("run_matrix")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round1.yaml", help="實驗設定檔")
    parser.add_argument("--no-resume", action="store_true", help="忽略 checkpoint，從頭執行（舊輸出改名為 .bak）")
    parser.add_argument("--save-processed", action="store_true", help="另存方法處理後的音訊（除錯用）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """執行 run_experiment()，結束後驗證輸出；回傳 0 代表成功且驗證通過。"""
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    if args.save_processed:
        cfg.save_processed = True
    set_seed(cfg.seed)
    out = run_experiment(cfg, resume=not args.no_resume)
    problems = validate_results(out)
    for p in problems:
        log.error(p)
    if not problems:
        log.info("輸出驗證通過：%s", out)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
