"""產生 / 讀取 / 驗證 manifest.csv（欄位見 dashboard/data_contract.md）。"""

from pathlib import Path

import pandas as pd

from library.data.types import Utterance

MANIFEST_COLUMNS: tuple[str, ...] = (
    "utt_id",
    "wav_path",
    "ref_raw",
    "speaker_id",
    "gender",
    "age",
    "audio_sec",
    "source_file",
)


def build_manifest(samples: pd.DataFrame, out_dir: Path) -> Path:
    """轉檔所有樣本並寫出 manifest.csv。

    依序指派 utt_id（utt001…）與匿名 speaker_id（spk01…），client_id 不寫入。

    參數:
        samples: sample_utterances() 的輸出。
        out_dir: 輸出資料夾（例：data/clean），WAV 與 manifest.csv 都寫在這裡。

    回傳:
        manifest.csv 路徑。

    例外:
        FileNotFoundError: 樣本中的來源音檔不存在。
    """
    raise NotImplementedError


def load_manifest(path: str | Path) -> list[Utterance]:
    """讀取 manifest.csv（utf-8）。

    參數:
        path: manifest.csv 路徑。

    回傳:
        Utterance 清單，順序與檔案相同；wav_path 解析為絕對路徑。

    例外:
        FileNotFoundError: manifest 不存在。
        ValueError: 缺少必要欄位。
    """
    raise NotImplementedError


def validate_manifest(path: str | Path) -> list[str]:
    """回傳問題清單（空清單代表通過）：檔案是否存在、取樣率、重複 ID 等。

    參數:
        path: manifest.csv 路徑。

    回傳:
        問題描述清單；空清單代表通過。
    """
    raise NotImplementedError
