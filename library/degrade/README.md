# library/degrade/

產生實驗用的劣化音訊。結果預先存檔，確保每個模型聽到相同音訊。

| 檔案 | 做什麼 |
|---|---|
| noise.py | add_noise()：把噪音縮放到指定 SNR 後疊加 |
| reverb.py | add_reverb()：用 pyroomacoustics 模擬房間殘響 |
| volume.py | change_volume()：音量增減（dB） |
| noise_only.py | make_noise_only()：切出純噪音片段（幻覺測試用） |
| registry.py | 條件名稱對應處理函式；apply_condition()、list_conditions() |

## 新增條件
1. 在 conditions.yaml 新增條件與參數
2. 若是新類型，在本資料夾新增處理檔並於 registry.py 註冊
