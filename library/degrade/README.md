# library/degrade/

產生實驗用的劣化音訊。結果預先存檔，確保每個模型聽到相同音訊。

| 檔案 | 做什麼 |
|---|---|
| noise.py | add_noise()：把噪音縮放到指定 SNR 後疊加；measure_snr()：由 clean 與加噪結果估計實際 SNR |
| reverb.py | add_reverb()：用 pyroomacoustics 模擬房間殘響（輸出 RMS 調回與輸入相同）；estimate_rt60() |
| volume.py | change_volume()：音量增減（dB） |
| noise_only.py | make_noise_only()：切出互不重疊的純噪音片段（幻覺測試用） |
| registry.py | 條件名稱對應處理函式；apply_condition()、build_noise_only_clips()、list_conditions()、噪音源管理 |

## 規則
- 噪音源每個檔案切兩段：前段用於疊加，最後 `noise_holdout_sec` 秒只給 noise_only，兩者不重疊。
- 同一句在各 SNR 條件使用相同 seed → 相同噪音環境與片段，只改變 SNR。
- 疊加或殘響後峰值超過 0.99 時整體等比例縮小（不改變 SNR），避免 16-bit 削波。
- noise_only 預設取自 holdout 段；條件設定 `segment: mix` 時改取疊加段（調參資料用，避免用到測試集的 noise_only 噪音）。
- 殘響的擺放位置由 `geometry_seed` 固定，所有句子使用同一個房間脈衝響應。

## 新增條件
1. 在 conditions.yaml 新增條件與參數
2. 若是新類型，在本資料夾新增處理檔並於 registry.py 註冊
