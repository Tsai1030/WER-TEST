"""library.data：TSV 讀取、抽樣、轉檔、manifest。

使用暫存資料夾中的合成資料（不讀 data/），任何機器都能執行。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from library.data import (
    MANIFEST_COLUMNS,
    build_manifest,
    convert_to_wav,
    get_duration,
    load_commonvoice_tsv,
    load_manifest,
    read_audio,
    sample_utterances,
    speakers_of,
    validate_manifest,
    write_noise_entries,
)

SRC_SR = 48000
HEADER = "client_id\tpath\tsentence_id\tsentence\tsentence_domain\tup_votes\tdown_votes\tage\tgender\taccents"


def _write_clip(path: Path, sec: float, channels: int = 1) -> None:
    n = int(sec * SRC_SR)
    t = np.arange(n) / SRC_SR
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    sf.write(path, np.stack([tone] * channels, axis=1) if channels > 1 else tone, SRC_SR)


def _make_dataset(root: Path, gender_raw: str, rows: list[tuple[str, str, float, int]]) -> Path:
    """rows: (client_id, sentence, 秒數, down_votes)。回傳 TSV 路徑。"""
    clips = root / "clips"
    clips.mkdir(parents=True)
    lines = [HEADER]
    for i, (client, sentence, sec, down) in enumerate(rows):
        name = f"common_voice_en_{gender_raw[0]}{i:04d}.wav"
        _write_clip(clips / name, sec)
        lines.append(f"{client}\t{name}\tsid{i}\t{sentence}\t\t2\t{down}\tthirties\t{gender_raw}\tUnited States English")
    tsv = root / "test.tsv"
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tsv


@pytest.fixture
def corpus(tmp_path: Path) -> pd.DataFrame:
    good = "one two three four five six"
    female_rows = [(f"f{s}", good, 4.0, 0) for s in range(6)] + [
        ("f0", good, 5.0, 0),  # 同一說話者第二句
        ("f9", "too short", 4.0, 0),  # 詞數不足
        ("f8", good, 9.5, 0),  # 太長
        ("f7", good, 4.0, 2),  # 有反對票
    ]
    male_rows = [(f"m{s}", good, 3.5, 0) for s in range(6)] + [("m9", good, 2.0, 0)]  # 太短
    tsv_f = _make_dataset(tmp_path / "female", "female_feminine", female_rows)
    tsv_m = _make_dataset(tmp_path / "male", "male_masculine", male_rows)
    return pd.concat(
        [
            load_commonvoice_tsv(tsv_f, tmp_path / "female" / "clips"),
            load_commonvoice_tsv(tsv_m, tmp_path / "male" / "clips"),
        ],
        ignore_index=True,
    )


def test_load_tsv_parses_quotes_and_gender(tmp_path: Path) -> None:
    sentence = '"He was praised for ""singing"" the anthem."'
    clips = tmp_path / "clips"
    clips.mkdir()
    _write_clip(clips / "a.wav", 1.0)
    tsv = tmp_path / "test.tsv"
    tsv.write_text(
        f"{HEADER}\nc1\ta.wav\ts1\t{sentence}\t\t2\t0\t\tmale_masculine\tUS\n", encoding="utf-8"
    )
    df = load_commonvoice_tsv(tsv, clips)
    assert df.loc[0, "sentence"] == 'He was praised for "singing" the anthem.'
    assert df.loc[0, "gender"] == "male"
    assert Path(df.loc[0, "path"]) == clips / "a.wav"
    assert df.loc[0, "source_file"] == "a.wav"
    assert df.loc[0, "down_votes"] == 0


def test_load_tsv_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_commonvoice_tsv(tmp_path / "nope.tsv", tmp_path)


def test_sample_rules(corpus: pd.DataFrame) -> None:
    s = sample_utterances(corpus, n_per_gender=5, min_sec=3.0, max_sec=8.0, min_words=5, seed=42)
    assert s["gender"].value_counts().to_dict() == {"female": 5, "male": 5}
    assert s["client_id"].is_unique
    assert s["audio_sec"].between(3.0, 8.0).all()
    assert (s["n_words"] >= 5).all()
    assert (s["down_votes"] == 0).all()
    assert not s["client_id"].isin(["f7", "f8", "f9", "m9"]).any()
    assert list(s["gender"]) == ["female"] * 5 + ["male"] * 5


def test_sample_is_deterministic(corpus: pd.DataFrame) -> None:
    a = sample_utterances(corpus, n_per_gender=5, seed=7)
    b = sample_utterances(corpus.sample(frac=1, random_state=1), n_per_gender=5, seed=7)
    assert list(a["path"]) == list(b["path"])


def test_sample_excludes_speakers(corpus: pd.DataFrame) -> None:
    """調參集：排除測試集說話者（以原始檔名反查 client_id）。"""
    test = sample_utterances(corpus, n_per_gender=2, seed=42)
    excluded = speakers_of(corpus, set(test["source_file"]))
    assert excluded == set(test["client_id"])
    tune = sample_utterances(corpus, n_per_gender=2, seed=42, exclude_speakers=excluded)
    assert not tune["client_id"].isin(excluded).any()


def test_sample_insufficient(corpus: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="不足"):
        sample_utterances(corpus, n_per_gender=7)


def test_convert_to_wav(tmp_path: Path) -> None:
    src = tmp_path / "stereo.wav"
    _write_clip(src, 2.0, channels=2)
    dst = tmp_path / "out" / "x.wav"
    sec = convert_to_wav(src, dst)
    info = sf.info(dst)
    assert (info.samplerate, info.channels) == (16000, 1)
    assert sec == pytest.approx(2.0, abs=1e-3)
    assert get_duration(dst) == pytest.approx(sec, abs=1e-3)
    wav, sr = read_audio(dst)
    assert wav.ndim == 1 and wav.dtype == np.float32 and sr == 16000


def test_build_load_validate(corpus: pd.DataFrame, tmp_path: Path) -> None:
    samples = sample_utterances(corpus, n_per_gender=5, seed=42)
    manifest = build_manifest(samples, tmp_path / "clean")

    df = pd.read_csv(manifest, encoding="utf-8", dtype=str, keep_default_na=False)
    assert tuple(df.columns) == MANIFEST_COLUMNS
    assert "client_id" not in df.columns
    assert not df.isin(samples["client_id"].tolist()).any().any()
    assert list(df["utt_id"]) == [f"utt{i:03d}" for i in range(1, 11)]
    assert list(df["speaker_id"]) == [f"spk{i:02d}" for i in range(1, 11)]
    assert validate_manifest(manifest) == []

    utts = load_manifest(manifest)
    assert len(utts) == 10
    assert all(u.wav_path.is_absolute() and u.wav_path.is_file() for u in utts)
    assert utts[0].ref_raw == "one two three four five six"


def test_validate_detects_problems(corpus: pd.DataFrame, tmp_path: Path) -> None:
    samples = sample_utterances(corpus, n_per_gender=5, seed=42)
    manifest = build_manifest(samples, tmp_path / "clean")
    df = pd.read_csv(manifest, encoding="utf-8", dtype=str, keep_default_na=False)
    df.loc[1, "utt_id"] = "utt001"  # 重複 ID
    df.loc[2, "wav_path"] = (tmp_path / "missing.wav").as_posix()  # 檔案不存在
    df.loc[3, "gender"] = "unknown"
    sf.write(tmp_path / "hi.wav", np.zeros(48000, dtype="float32"), 48000)
    df.loc[4, "wav_path"] = (tmp_path / "hi.wav").as_posix()  # 取樣率錯誤
    df.to_csv(manifest, index=False, encoding="utf-8")

    problems = "\n".join(validate_manifest(manifest))
    assert "utt_id 重複" in problems
    assert "音檔不存在" in problems
    assert "gender 不合法" in problems
    assert "取樣率 48000" in problems


def test_write_noise_entries_replaces_noise_rows(corpus: pd.DataFrame, tmp_path: Path) -> None:
    samples = sample_utterances(corpus, n_per_gender=5, seed=42)
    manifest = build_manifest(samples, tmp_path / "clean")
    clips = tmp_path / "noise_only"
    clips.mkdir()
    entries = []
    for k in (1, 2):
        wav = clips / f"noise{k:02d}.wav"
        sf.write(wav, np.zeros(16000 * 2, dtype="float32"), 16000)
        entries.append({"utt_id": f"noise{k:02d}", "wav_path": wav, "audio_sec": 2.0, "source_file": "ENVA"})

    write_noise_entries(manifest, entries)
    write_noise_entries(manifest, entries)  # 重跑不可重複附加
    df = pd.read_csv(manifest, encoding="utf-8", dtype=str, keep_default_na=False)
    assert len(df) == 12
    assert list(df["utt_id"][-2:]) == ["noise01", "noise02"]
    assert (df.loc[df["utt_id"].str.startswith("noise"), ["ref_raw", "speaker_id", "gender"]] == "").all().all()
    assert validate_manifest(manifest) == []
    with pytest.raises(ValueError):
        write_noise_entries(manifest, [{**entries[0], "utt_id": "n1"}])


def test_validate_missing_manifest(tmp_path: Path) -> None:
    assert validate_manifest(tmp_path / "manifest.csv") != []
