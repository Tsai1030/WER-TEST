"""依規則抽樣（性別、時長、詞數、每人一句）。"""

import pandas as pd


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
        抽樣結果 DataFrame，新增 audio_sec 欄。

    例外:
        ValueError: 符合條件的句子不足 n_per_gender 句。
    """
    raise NotImplementedError
