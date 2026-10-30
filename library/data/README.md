# library/data/

資料集讀取、抽樣、轉檔與 manifest 管理。

| 檔案 | 做什麼 |
|---|---|
| types.py | Utterance 資料結構（utt_id、wav_path、ref_raw、speaker_id、gender、audio_sec） |
| commonvoice.py | load_commonvoice_tsv()：讀 Common Voice TSV，整理成 DataFrame |
| sampler.py | sample_utterances()：依性別、時長、詞數、每人一句抽樣，固定 seed |
| audio_io.py | read_audio()、write_audio()、convert_to_wav()：統一轉 16kHz mono |
| manifest.py | build_manifest()、load_manifest()、validate_manifest() |

## 範例
    df = load_commonvoice_tsv("data/raw/female/test.tsv", "data/raw/female/clips")
    samples = sample_utterances(df, n_per_gender=25, seed=42)
    build_manifest(samples, Path("data/clean"))

## 注意
- speaker_id 一律匿名化（spk01…），不輸出 Common Voice client_id
- 新增資料集：新增一個讀取檔（例如 mslt.py），輸出相同欄位的 DataFrame 即可
