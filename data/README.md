# data/

不進版控。Common Voice 條款禁止重新散佈，請勿上傳到公開位置。

| 資料夾 | 內容 |
|---|---|
| raw/archives/ | 手動下載的原始 .tar.gz（保留） |
| raw/female/、raw/male/ | 只解出的 test split：test.tsv、clip_list.txt、clips/ |
| noise/source/ | 噪音源：DEMAND 16kHz（CC BY 4.0，doi:10.5281/zenodo.1227121）DKITCHEN、DLIVING、DWASHING、OOFFICE 各取 ch01，各 300 秒 |
| clean/ | 50 句 16kHz mono WAV + manifest.csv（含 noise01–noise10 列，指向 degraded/noise_only/） |
| degraded/ | degraded/<condition>/<utt_id>.wav；degraded/noise_only/noiseNN.wav；degrade_log.csv（逐檔驗證紀錄） |

## 重新產生
    uv run python scripts/01_prepare_data.py --config configs/round1.yaml
    uv run python scripts/02_make_degraded.py --config configs/round1.yaml

噪音源下載：Zenodo 記錄 1227121 的 `<ENV>_16k.zip`，驗證 md5 後只取 `<ENV>/ch01.wav` 存成 `noise/source/<ENV>.wav`
（md5 見 docs/decisions.md）。
