"""抽樣 + 轉檔 + 產生 manifest。

用法:
    uv run python scripts/01_prepare_data.py --config configs/round1.yaml
"""

import argparse

import pandas as pd

from library.data import (
    build_manifest,
    load_commonvoice_tsv,
    sample_utterances,
    validate_manifest,
)
from library.utils import get_logger, load_yaml, resolve_path, set_seed, setup_logging

log = get_logger("prepare_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round1.yaml", help="實驗設定檔")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """抽樣 50 句、轉成 16kHz mono WAV、寫出 data/clean/manifest.csv；回傳 0 代表成功。"""
    args = parse_args(argv)
    setup_logging()
    cfg = load_yaml(resolve_path(args.config))
    seed = cfg["seed"]
    set_seed(seed)
    sampling = cfg["sampling"]

    frames = []
    for gender, raw in cfg["paths"]["raw"].items():
        raw_dir = resolve_path(raw)
        df = load_commonvoice_tsv(raw_dir / sampling["tsv_name"], raw_dir / "clips")
        log.info("%s：讀入 %d 句、%d 位說話者", gender, len(df), df["client_id"].nunique())
        frames.append(df)

    samples = sample_utterances(
        pd.concat(frames, ignore_index=True),
        n_per_gender=sampling["n_per_gender"],
        min_sec=sampling["min_sec"],
        max_sec=sampling["max_sec"],
        min_words=sampling["min_words"],
        max_down_votes=sampling["max_down_votes"],
        one_per_speaker=sampling["one_per_speaker"],
        seed=seed,
    )

    manifest_path = resolve_path(cfg["paths"]["manifest"])
    written = build_manifest(samples, manifest_path.parent)
    if written != manifest_path:
        log.warning("manifest 實際寫入 %s（設定檔為 %s）", written, manifest_path)

    by_gender = samples["gender"].value_counts().to_dict()
    log.info(
        "抽樣完成：%d 句 %s、%d 位說話者、時長 %.2f–%.2f 秒（平均 %.2f）、詞數 %d–%d",
        len(samples),
        by_gender,
        samples["client_id"].nunique(),
        samples["audio_sec"].min(),
        samples["audio_sec"].max(),
        samples["audio_sec"].mean(),
        samples["n_words"].min(),
        samples["n_words"].max(),
    )

    problems = validate_manifest(written)
    for p in problems:
        log.error(p)
    if problems:
        return 1
    log.info("manifest 驗證通過：%s", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
