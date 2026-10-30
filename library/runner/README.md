# library/runner/

串接所有模組，執行「模型 × 方法 × 條件 × 句子」實驗矩陣。

| 檔案 | 做什麼 |
|---|---|
| config.py | load_config()：讀 YAML 並驗證欄位 |
| matrix.py | iter_matrix()：依「模型 → 方法 → 條件 → 句子」順序展開 |
| checkpoint.py | 記錄已完成組合，重跑時跳過 |
| runner.py | run_experiment()：主流程，逐列寫入結果 |

## 流程
載入模型 → 每個方法重設資源統計 → 每個條件讀取劣化音檔 → 方法處理 → 模型辨識
→ 正規化 → 對齊 → 寫入一列 → 該模型結束後 unload
