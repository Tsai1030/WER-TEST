"""產生所有劣化條件音檔（data/degraded/<condition>/<utt_id>.wav）。

用法:
    uv run python scripts/02_make_degraded.py --config configs/round1.yaml

需先執行 01_prepare_data.py；重跑 01 會覆寫 manifest（含 noise 列），之後需重跑本腳本。
"""

import argparse

from library.data import validate_manifest
from library.runner import make_degraded_set
from library.utils import get_logger, load_yaml, resolve_path, set_seed, setup_logging

log = get_logger("make_degraded")

SNR_TOL_DB = 0.5  # plan.md M2 驗收：實測 SNR 與目標誤差 < 0.5 dB
GAIN_TOL_DB = 0.1
RT60_TOL_SEC = 0.15
NOISE_RMS_TOL_DB = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round1.yaml", help="實驗設定檔")
    parser.add_argument("--conditions", default="configs/conditions.yaml", help="劣化條件設定檔")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """產生設定檔 conditions 列出的劣化條件與純噪音片段；回傳 0 代表成功且驗證通過。"""
    args = parse_args(argv)
    setup_logging()
    cfg = load_yaml(resolve_path(args.config))
    cond_cfg = load_yaml(resolve_path(args.conditions))
    set_seed(cfg["seed"])
    manifest = resolve_path(cfg["paths"]["manifest"])
    out_root = resolve_path(cfg["paths"]["degraded_dir"])

    result = make_degraded_set(manifest, out_root, cfg["conditions"], cond_cfg, cfg["seed"])

    tolerance = {"noise": SNR_TOL_DB, "volume": GAIN_TOL_DB, "reverb": RT60_TOL_SEC,
                 "noise_only": NOISE_RMS_TOL_DB, "clean": 1e-4}
    failed = 0
    print(f"{'condition':11s} {'type':10s} {'n':>3s} {'target':>7s} {'measured(mean)':>15s} {'max|err|':>9s}  結果")
    for cond, g in result.groupby("condition", sort=False):
        kind = g["type"].iloc[0]
        max_err = float(g["error"].abs().max())
        ok = max_err <= tolerance[kind]
        failed += int(not ok)
        print(f"{cond:11s} {kind:10s} {len(g):3d} {g['target'].iloc[0]:7.2f} {g['measured'].mean():15.3f} "
              f"{max_err:9.3f}  {'OK' if ok else 'FAIL'}")
    envs = result[result["type"] == "noise"].drop_duplicates("utt_id")["noise_env"].value_counts()
    print("疊加用噪音環境（每句一個，各 SNR 共用）：", envs.to_dict())
    print("純噪音片段來源：", result[result["type"] == "noise_only"]["noise_env"].value_counts().to_dict())

    problems = validate_manifest(manifest)
    for p in problems:
        log.error(p)
    log.info("輸出：%s（驗證紀錄 degrade_log.csv）", out_root)
    return 1 if failed or problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
