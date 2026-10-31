# configs/

所有實驗參數集中在這裡，程式不寫死任何參數。

| 檔案 | 用途 |
|---|---|
| round1.yaml | 第一輪總設定：要跑哪些模型、方法、條件，輸出到哪 |
| round2.yaml | 第二輪：負 SNR（snrm05 / snrm10）、loudnorm、較溫和的 denoise（prop_decrease 0.5） |
| round3.yaml | 第三輪：VAD 開關用法（`mode: gate`、threshold 0.3）的測試集驗證，條件同 Round 2 |
| vad_tune.yaml | VAD 參數掃描（Round 3 前置）：調參資料（排除測試集說話者）、掃描的 VAD 設定、使用模型 |
| conditions.yaml | 劣化條件定義（SNR、殘響、音量、純噪音） |
| conditions_tune.yaml | 調參資料用的條件（clean、snrm05、snrm10；noise_only 取自噪音疊加段，不用 holdout） |
| models/*.yaml | 每個模型的 HF ID、框架、解碼參數 |
| methods/*.yaml | 每個前處理方法的預設參數 |

## 新增一輪實驗
複製 round1.yaml 為 round2.yaml，修改 name 與 output_dir，其餘依需要調整。
某輪要改方法參數時，在該輪設定檔寫 `method_params`（取代 methods/*.yaml，只作用於該輪），
不要直接改 methods/*.yaml，以免前一輪無法重現。
