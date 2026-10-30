"""讀寫音檔、MP3 → 16kHz mono WAV。"""

from pathlib import Path

import numpy as np


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """讀取音檔為 float32 一維陣列。

    參數:
        path: 音檔路徑。

    回傳:
        (wav, sr)：wav 為 float32、範圍 [-1, 1] 的一維陣列；sr 為取樣率。

    例外:
        FileNotFoundError: 音檔不存在。
    """
    raise NotImplementedError


def write_audio(path: str | Path, wav: np.ndarray, sr: int = 16000) -> None:
    """將一維 float32 陣列寫成 16-bit PCM WAV；父資料夾不存在時自動建立。

    參數:
        path: 輸出路徑。
        wav: 音訊陣列。
        sr: 取樣率。

    例外:
        ValueError: wav 不是一維陣列。
    """
    raise NotImplementedError


def get_duration(mp3_path: Path) -> float:
    """讀取音檔時長（秒）。

    參數:
        mp3_path: 音檔路徑（MP3 或 WAV）。

    回傳:
        時長（秒）。

    例外:
        FileNotFoundError: 音檔不存在。
    """
    raise NotImplementedError


def convert_to_wav(src: Path, dst: Path, sr: int = 16000, mono: bool = True) -> float:
    """轉檔並回傳音檔秒數。

    參數:
        src: 來源音檔（通常為 Common Voice MP3）。
        dst: 輸出 WAV 路徑。
        sr: 目標取樣率。
        mono: 是否轉為單聲道。

    回傳:
        輸出音檔秒數。

    例外:
        FileNotFoundError: 來源音檔不存在。
    """
    raise NotImplementedError
