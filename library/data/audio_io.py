"""讀寫音檔、MP3 → 16kHz mono WAV。

MP3 由 soundfile（libsndfile ≥ 1.1）直接解碼，不需要 ffmpeg；重取樣用 librosa（soxr）。
"""

from pathlib import Path

import numpy as np
import soundfile as sf


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"音檔不存在：{path}")


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """讀取音檔為 float32 一維陣列。

    多聲道音檔會平均成單聲道。

    參數:
        path: 音檔路徑。

    回傳:
        (wav, sr)：wav 為 float32、範圍 [-1, 1] 的一維陣列；sr 為取樣率。

    例外:
        FileNotFoundError: 音檔不存在。
    """
    p = Path(path)
    _require_file(p)
    wav, sr = sf.read(p, dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def write_audio(path: str | Path, wav: np.ndarray, sr: int = 16000) -> None:
    """將一維 float32 陣列寫成 16-bit PCM WAV；父資料夾不存在時自動建立。

    參數:
        path: 輸出路徑。
        wav: 音訊陣列。
        sr: 取樣率。

    例外:
        ValueError: wav 不是一維陣列。
    """
    if wav.ndim != 1:
        raise ValueError(f"wav 必須是一維陣列，實際為 {wav.ndim} 維")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")


def get_duration(mp3_path: Path) -> float:
    """讀取音檔時長（秒）。

    只讀標頭資訊，不解碼整個檔案（MP3 每檔約 5ms）。

    參數:
        mp3_path: 音檔路徑（MP3 或 WAV）。

    回傳:
        時長（秒）。

    例外:
        FileNotFoundError: 音檔不存在。
    """
    p = Path(mp3_path)
    _require_file(p)
    return float(sf.info(p).duration)


def convert_to_wav(src: Path, dst: Path, sr: int = 16000, mono: bool = True) -> float:
    """轉檔並回傳音檔秒數。

    參數:
        src: 來源音檔（通常為 Common Voice MP3）。
        dst: 輸出 WAV 路徑（16-bit PCM）。
        sr: 目標取樣率。
        mono: 是否轉為單聲道。

    回傳:
        輸出音檔秒數。

    例外:
        FileNotFoundError: 來源音檔不存在。
    """
    src = Path(src)
    _require_file(src)
    wav, src_sr = sf.read(src, dtype="float32", always_2d=True)  # (frames, channels)
    if mono:
        wav = wav.mean(axis=1, keepdims=True)
    if src_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=src_sr, target_sr=sr, res_type="soxr_hq", axis=0)
    out = wav[:, 0] if mono else wav
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, np.clip(out, -1.0, 1.0), sr, subtype="PCM_16")
    return len(out) / sr
