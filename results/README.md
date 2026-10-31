# results/

每輪實驗一個資料夾，例如 round1/。

| 檔案 | 內容 |
|---|---|
| results.csv | 逐句結果（每列一個 模型×條件×方法×句子） |
| resources.csv | 每個 模型×方法 的峰值記憶體、模型大小、載入時間 |
| meta.json | 實驗環境與設定 |
| checkpoint.json | 續跑用，不需交給網頁 |

vad_tune/ 是 VAD 參數掃描（06_vad_sweep.py）的輸出：sweep.csv 每列一個 設定 × 條件 × 句子（S/D/I/N、保留秒數 kept_sec、純噪音是否放行 passed、正規化後轉寫），不給網頁使用。

round4/ 是喚醒詞與講完判斷（08_wake_eval.py）的輸出，格式與網頁不同、不進入網頁的自動分組（網頁的 Round 4 區塊只放彙總數字）：

| 檔案 | 內容 |
|---|---|
| kws_negative.csv | 每個 喚醒詞模型 × 門檻 × 來源（cv_en 英文語音 / noise 噪音）的音訊長度與誤喚醒次數 |
| kws_negative_hits.csv | 每次誤喚醒：來源、句子編號、句中時間、句子原文 |
| endpoint.csv | 每個 模擬情境 × 講完判斷參數：判斷結果、判斷時間、實際語音範圍、是否截斷、等待時間 |
| kws_positive.csv | 自錄音檔（正樣本 × 噪音、音近詞）的偵測次數；音檔本身不放這裡 |
| pipeline.csv | 整段流程：每個 測試集情境 × 講完判斷（含 oracle）：判斷結果、是否截斷、等待時間、辨識時間、講完到出字、轉寫與 S/D/I/N |
| standby.csv | 待機成本：每個元件（喚醒詞模型、VAD、ASR）× 音訊（語音 / 噪音）的 CPU 佔用、每塊處理時間、載入後增加的記憶體 |

欄位規格見 dashboard/data_contract.md。
