"""依規則抽樣（性別、時長、詞數、每人一句）。"""

from pathlib import Path

import numpy as np
import pandas as pd

from library.data.audio_io import get_duration

_GENDERS = ("female", "male")


def sample_utterances(
    df: pd.DataFrame,
    n_per_gender: int = 25,
    min_sec: float = 3.0,
    max_sec: float = 8.0,
    min_words: int = 5,
    max_down_votes: int = 0,
    one_per_speaker: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """依規則抽樣；同一 seed 必須得到相同結果。

    TSV 沒有時長欄位，時長需讀音檔取得。為避免讀完上萬個 MP3：
    先依詞數、down_votes、每人一句篩選並以 seed 打亂，
    再依序讀取時長，湊滿 n_per_gender 句就停止。

    打亂前先依 path 排序，結果與 TSV 列順序無關。某位說話者的候選句時長不符時，
    會繼續嘗試同一人的下一句，直到湊滿或候選用完。

    參數:
        df: load_commonvoice_tsv() 的輸出（可為女聲與男聲合併後的 DataFrame）。
        n_per_gender: 每個性別抽幾句。
        min_sec: 最短秒數（含）。
        max_sec: 最長秒數（含）。
        min_words: 最少詞數（以空白切詞）。
        max_down_votes: down_votes 上限。
        one_per_speaker: 每位說話者最多一句。
        seed: 亂數種子。

    回傳:
        抽樣結果 DataFrame（先女後男，各自依抽中順序），新增 audio_sec 與 n_words 欄。

    例外:
        ValueError: 符合條件的句子不足 n_per_gender 句。
    """
    work = df.copy()
    work["n_words"] = work["sentence"].str.split().str.len()
    mask = (
        work["gender"].isin(_GENDERS)
        & (work["n_words"] >= min_words)
        & (work["down_votes"] <= max_down_votes)
    )
    work = work[mask].sort_values("path", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    work = work.iloc[rng.permutation(len(work))]

    picked: dict[str, list[dict]] = {g: [] for g in _GENDERS}
    used_speakers: set[str] = set()
    for row in work.to_dict("records"):
        bucket = picked[row["gender"]]
        if len(bucket) >= n_per_gender:
            continue
        if one_per_speaker and row["client_id"] in used_speakers:
            continue
        sec = get_duration(Path(row["path"]))
        if not min_sec <= sec <= max_sec:
            continue
        bucket.append({**row, "audio_sec": sec})
        used_speakers.add(row["client_id"])
        if all(len(picked[g]) >= n_per_gender for g in _GENDERS):
            break

    short = {g: len(picked[g]) for g in _GENDERS if len(picked[g]) < n_per_gender}
    if short:
        raise ValueError(f"符合條件的句子不足 {n_per_gender} 句：{short}")
    return pd.DataFrame(picked["female"] + picked["male"]).reset_index(drop=True)
