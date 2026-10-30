"""讀取 Common Voice TSV。"""

from pathlib import Path

import pandas as pd

_REQUIRED_COLUMNS = ("client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender")
_GENDER_MAP = {
    "female_feminine": "female",
    "male_masculine": "male",
    "female": "female",
    "male": "male",
}


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
        DataFrame，至少含 path、sentence、client_id、gender、up_votes、down_votes、age 欄，
        另加 source_file（原始 MP3 檔名）。up_votes / down_votes 為 int；
        無法辨識的 gender 轉為空字串。

    例外:
        FileNotFoundError: TSV 或 clips 資料夾不存在。
        ValueError: TSV 缺少必要欄位。
    """
    tsv_path, clips_dir = Path(tsv_path), Path(clips_dir)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"TSV 不存在：{tsv_path}")
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"clips 資料夾不存在：{clips_dir}")

    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"TSV 缺少欄位 {missing}：{tsv_path}")

    # path 欄為檔名（不含 clips/）；保險起見兩種寫法都接受
    df["source_file"] = df["path"].map(lambda p: Path(p).name)
    df["path"] = df["source_file"].map(lambda name: str(clips_dir / name))
    df["gender"] = df["gender"].map(lambda g: _GENDER_MAP.get(g, ""))
    for col in ("up_votes", "down_votes"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df
