# configs/

所有實驗參數集中在這裡，程式不寫死任何參數。

| 檔案 | 用途 |
|---|---|
| round1.yaml | 第一輪總設定：要跑哪些模型、方法、條件，輸出到哪 |
| conditions.yaml | 劣化條件定義（SNR、殘響、音量、純噪音） |
| models/*.yaml | 每個模型的 HF ID、框架、解碼參數 |
| methods/*.yaml | 每個前處理方法的參數 |

## 新增一輪實驗
複製 round1.yaml 為 round2.yaml，修改 name 與 output_dir，其餘依需要調整。
