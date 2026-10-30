# results/

每輪實驗一個資料夾，例如 round1/。

| 檔案 | 內容 |
|---|---|
| results.csv | 逐句結果（每列一個 模型×條件×方法×句子） |
| resources.csv | 每個 模型×方法 的峰值記憶體、模型大小、載入時間 |
| meta.json | 實驗環境與設定 |
| checkpoint.json | 續跑用，不需交給網頁 |

欄位規格見 dashboard/data_contract.md。
