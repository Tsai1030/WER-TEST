"""展開「模型 × 方法 × 條件 × 句子」組合。"""

from collections.abc import Iterator
from pathlib import Path

from library.data import Utterance, load_manifest
from library.runner.config import ExperimentConfig


def split_utterances(cfg: ExperimentConfig) -> tuple[list[Utterance], list[Utterance]]:
    """由 manifest 取出（語音句, 純噪音片段）。

    設定 max_utterances 時，依 manifest 順序女、男輪流取，讓小規模測試也維持性別平衡；
    回傳的語音句仍依 manifest 順序排列。

    例外:
        FileNotFoundError: manifest 不存在。
    """
    utts = load_manifest(cfg.manifest)
    speech = [u for u in utts if u.utt_id.startswith("utt")]
    noise = [u for u in utts if u.utt_id.startswith("noise")]
    if cfg.max_utterances is not None and cfg.max_utterances < len(speech):
        by_gender = [[u for u in speech if u.gender == g] for g in ("female", "male")]
        rest = [u for u in speech if u.gender not in ("female", "male")]
        picked = []
        for i in range(len(speech)):
            for group in (*by_gender, rest):
                if i < len(group) and len(picked) < cfg.max_utterances:
                    picked.append(group[i])
        keep = {u.utt_id for u in picked}
        speech = [u for u in speech if u.utt_id in keep]
    return speech, noise


def degraded_path(cfg: ExperimentConfig, condition: str, utt_id: str) -> Path:
    """劣化音檔路徑：degraded_dir/<condition>/<utt_id>.wav。"""
    return cfg.degraded_dir / condition / f"{utt_id}.wav"


def iter_matrix(cfg: ExperimentConfig) -> Iterator[tuple[str, str, str, Utterance]]:
    """依「模型 → 方法 → 條件 → 句子」順序展開（最外層為模型，模型只載入一次）。

    noise_only 條件使用純噪音片段（noise01…）而非 manifest 中的語音句。

    參數:
        cfg: 實驗設定。

    回傳:
        依序產生 (model, method, condition, utterance)。

    例外:
        FileNotFoundError: manifest 或劣化音檔資料夾不存在。
    """
    speech, noise = split_utterances(cfg)
    for cond in cfg.conditions:
        if not (cfg.degraded_dir / cond).is_dir():
            raise FileNotFoundError(f"劣化音檔資料夾不存在：{cfg.degraded_dir / cond}（請先執行 02_make_degraded.py）")
    for model in cfg.models:
        for method in cfg.methods:
            for cond in cfg.conditions:
                for u in noise if cond == "noise_only" else speech:
                    yield model, method, cond, u
