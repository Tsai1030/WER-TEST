"""產生 / 讀取 / 驗證 manifest.csv（欄位見 dashboard/data_contract.md）。"""

import re
from pathlib import Path

import pandas as pd
import soundfile as sf

from library.data.audio_io import convert_to_wav
from library.data.types import Utterance
from library.utils.paths import project_root, resolve_path

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
MANIFEST_NAME = "manifest.csv"

_SPEECH_ID = re.compile(r"utt\d{3}")
_NOISE_ID = re.compile(r"noise\d{2}")
_SPEAKER_ID = re.compile(r"spk\d{2,}")
_DURATION_TOL_SEC = 0.01


def _to_manifest_path(path: Path) -> str:
    """專案內的檔案寫成相對於專案根目錄的 posix 路徑；專案外（例如測試暫存檔）寫絕對路徑。"""
    path = path.resolve()
    try:
        return path.relative_to(project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(samples: pd.DataFrame, out_dir: Path) -> Path:
    """轉檔所有樣本並寫出 manifest.csv。

    依序指派 utt_id（utt001…）與匿名 speaker_id（spk01…，依首次出現順序），client_id 不寫入。

    參數:
        samples: sample_utterances() 的輸出。
        out_dir: 輸出資料夾（例：data/clean），WAV 與 manifest.csv 都寫在這裡。

    回傳:
        manifest.csv 路徑。

    例外:
        FileNotFoundError: 樣本中的來源音檔不存在。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    speaker_ids: dict[str, str] = {}
    rows = []
    for i, s in enumerate(samples.to_dict("records"), start=1):
        utt_id = f"utt{i:03d}"
        speaker_id = speaker_ids.setdefault(s["client_id"], f"spk{len(speaker_ids) + 1:02d}")
        wav_path = out_dir / f"{utt_id}.wav"
        sec = convert_to_wav(Path(s["path"]), wav_path)
        rows.append(
            {
                "utt_id": utt_id,
                "wav_path": _to_manifest_path(wav_path),
                "ref_raw": s["sentence"],
                "speaker_id": speaker_id,
                "gender": s["gender"],
                "age": s.get("age", ""),
                "audio_sec": round(sec, 4),
                "source_file": s.get("source_file") or Path(s["path"]).name,
            }
        )
    manifest = out_dir / MANIFEST_NAME
    pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)).to_csv(
        manifest, index=False, encoding="utf-8", lineterminator="\n"
    )
    return manifest


def write_noise_entries(path: str | Path, entries: list[dict]) -> None:
    """把純噪音片段寫入 manifest：先移除既有的 noise 列，再附加 entries（語音列不變）。

    參數:
        path: manifest.csv 路徑。
        entries: 每段一個 dict，含 utt_id（noise01…）、wav_path（Path）、audio_sec、source_file
            （噪音環境名稱）；ref_raw、speaker_id、gender、age 留空。

    例外:
        FileNotFoundError: manifest 不存在。
        ValueError: utt_id 不是 noiseNN 格式。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest 不存在：{path}")
    bad = [e["utt_id"] for e in entries if not _NOISE_ID.fullmatch(e["utt_id"])]
    if bad:
        raise ValueError(f"純噪音 utt_id 需為 noiseNN：{bad}")
    df = _read_manifest(path)
    speech = df[~df["utt_id"].str.fullmatch(_NOISE_ID.pattern)]
    rows = [
        {
            "utt_id": e["utt_id"],
            "wav_path": _to_manifest_path(Path(e["wav_path"])),
            "ref_raw": "",
            "speaker_id": "",
            "gender": "",
            "age": "",
            "audio_sec": round(float(e["audio_sec"]), 4),
            "source_file": e["source_file"],
        }
        for e in entries
    ]
    out = pd.concat([speech, pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))], ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def manifest_source_files(path: str | Path) -> set[str]:
    """回傳 manifest 中語音列的原始檔名（source_file 欄，不含 noise 列），供建立調參集時排除。"""
    df = _read_manifest(resolve_path(path))
    return set(df.loc[df["utt_id"].str.startswith("utt"), "source_file"])


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
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest 不存在：{path}")
    df = _read_manifest(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest 缺少欄位 {missing}：{path}")
    return [
        Utterance(
            utt_id=r["utt_id"],
            wav_path=resolve_path(r["wav_path"]),
            ref_raw=r["ref_raw"],
            speaker_id=r["speaker_id"],
            gender=r["gender"],
            audio_sec=float(r["audio_sec"]),
        )
        for r in df.to_dict("records")
    ]


def validate_manifest(path: str | Path, expected_sr: int = 16000) -> list[str]:
    """回傳問題清單（空清單代表通過）：檔案是否存在、取樣率、重複 ID 等。

    檢查項目：欄位齊全、utt_id 格式與唯一性、語音列的 ref_raw / speaker_id / gender、
    純噪音列的 speaker_id / gender 為空、音檔存在、取樣率、單聲道、audio_sec 與實際長度一致。

    參數:
        path: manifest.csv 路徑。
        expected_sr: 預期取樣率。

    回傳:
        問題描述清單；空清單代表通過。
    """
    path = Path(path)
    if not path.is_file():
        return [f"manifest 不存在：{path}"]
    df = _read_manifest(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        return [f"缺少欄位：{missing}"]

    problems: list[str] = []
    dup = df["utt_id"][df["utt_id"].duplicated()].unique().tolist()
    if dup:
        problems.append(f"utt_id 重複：{dup}")

    for r in df.to_dict("records"):
        uid = r["utt_id"]
        if _SPEECH_ID.fullmatch(uid):
            if not r["ref_raw"].strip():
                problems.append(f"{uid}：ref_raw 為空")
            if not _SPEAKER_ID.fullmatch(r["speaker_id"]):
                problems.append(f"{uid}：speaker_id 格式不符（{r['speaker_id']!r}）")
            if r["gender"] not in ("female", "male"):
                problems.append(f"{uid}：gender 不合法（{r['gender']!r}）")
        elif _NOISE_ID.fullmatch(uid):
            if r["speaker_id"] or r["gender"]:
                problems.append(f"{uid}：純噪音的 speaker_id / gender 應留空")
        else:
            problems.append(f"utt_id 格式不符：{uid!r}")

        wav = resolve_path(r["wav_path"])
        if not wav.is_file():
            problems.append(f"{uid}：音檔不存在（{r['wav_path']}）")
            continue
        info = sf.info(wav)
        if info.samplerate != expected_sr:
            problems.append(f"{uid}：取樣率 {info.samplerate}，應為 {expected_sr}")
        if info.channels != 1:
            problems.append(f"{uid}：聲道數 {info.channels}，應為 1")
        try:
            sec = float(r["audio_sec"])
        except ValueError:
            problems.append(f"{uid}：audio_sec 不是數字（{r['audio_sec']!r}）")
            continue
        if abs(sec - info.duration) > _DURATION_TOL_SEC:
            problems.append(f"{uid}：audio_sec {sec} 與實際長度 {info.duration:.4f} 不符")
    return problems
