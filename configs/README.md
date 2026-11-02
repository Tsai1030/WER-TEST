# configs/

所有實驗參數集中在這裡，程式不寫死任何參數。

| 檔案 | 用途 |
|---|---|
| round1.yaml | 第一輪總設定：要跑哪些模型、方法、條件，輸出到哪 |
| round2.yaml | 第二輪：負 SNR（snrm05 / snrm10）、loudnorm、較溫和的 denoise（prop_decrease 0.5） |
| round3.yaml | 第三輪：VAD 開關用法（`mode: gate`、threshold 0.3）的測試集驗證，與解碼方法 beam（qwen）/ greedy（whisper）；條件同 Round 2 |
| jetson_round3.yaml | Jetson Orin Nano 複驗：與 round3.yaml 相同的模型、方法與 50 句，`device: cuda`、fp32，條件為全部 10 種；輸出 `results/jetson/round3/` |
| round4.yaml | 第四輪：喚醒詞（小咪小咪，sherpa-onnx KWS 兩個模型 × 門檻）、講完判斷參數掃描與模擬情境、錄音清單、手動實測（live）、整段流程（pipeline）與待機成本（standby）設定 |
| vad_tune.yaml | VAD 參數掃描（Round 3 前置）：調參資料（排除測試集說話者）、掃描的 VAD 設定、使用模型 |
| conditions.yaml | 劣化條件定義（SNR、殘響、音量、純噪音） |
| conditions_tune.yaml | 調參資料用的條件（clean、snrm05、snrm10；noise_only 取自噪音疊加段，不用 holdout） |
| models/*.yaml | 每個模型的 HF ID、框架、解碼參數 |
| methods/*.yaml | 每個方法的預設參數（含解碼方法 beam.yaml、greedy.yaml） |

## 新增一輪實驗
複製 round1.yaml 為 round2.yaml，修改 name 與 output_dir，其餘依需要調整。
某輪要改方法參數時，在該輪設定檔寫 `method_params`（取代 methods/*.yaml，只作用於該輪），
不要直接改 methods/*.yaml，以免前一輪無法重現。
