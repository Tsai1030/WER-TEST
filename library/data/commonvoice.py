"""讀取 Common Voice TSV。"""

from pathlib import Path

import pandas as pd


def load_commonvoice_tsv(tsv_path: str | Path, clips_dir: str | Path) -> pd.DataFrame:
    """讀取 Common Voice TSV，回傳含 path、sentence、client_id、gender 的 DataFrame。

    - 必須用 pandas.read_csv(sep="\\t", encoding="utf-8") 等 CSV 解析器讀取：
      sentence 欄位含雙引號跳脫（例："...for ""singing"" the..."），不可直接用 split。
    - gender 原始值為 female_feminine / male_masculine，轉為 female / male。
    - client_id 為長雜湊，不輸出；由 build_manifest 轉成匿名 speaker_id。

    參數:
        tsv_path: TSV 檔路徑（例：data/raw/female/test.tsv）。
        clips_dir: 音檔資料夾（例：data/raw/female/clips）；path 欄會轉為此資料夾下的完整路徑。

    回傳:
        DataFrame，至少含 path、sentence、client_id、gender、up_votes、down_votes、age 欄。

    例外:
        FileNotFoundError: TSV 或 clips 資料夾不存在。
        ValueError: TSV 缺少必要欄位。
    """
    raise NotImplementedError
