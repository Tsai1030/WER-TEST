# library/runner/

串接所有模組，執行「模型 × 方法 × 條件 × 句子」實驗矩陣。

| 檔案 | 做什麼 |
|---|---|
| config.py | load_config()：讀 YAML 並驗證欄位 |
| matrix.py | iter_matrix()：依「模型 → 方法 → 條件 → 句子」順序展開 |
| checkpoint.py | 記錄已完成組合，重跑時跳過 |
| runner.py | run_experiment()：主流程，逐列寫入結果 |
| bench.py | bench_model()：M0 速度實測（暖機 1 句 + 計時 N 句），供 check_env.py 使用 |
| degrade_set.py | make_degraded_set()：M2 產生所有劣化音檔與純噪音片段，逐檔驗證並寫 degrade_log.csv |
| report.py | 05_quick_report.py 的計算：整體 WER、ΔWER 與 CI、幻覺率、正規化影響、男女聲、資源、一句話結論（定義同 dashboard/sections.md） |

## 流程
載入模型 → 每個方法重設資源統計 → 每個條件讀取劣化音檔 → 方法處理 → 模型辨識
→ 正規化 → 對齊 → 寫入一列 → 該模型結束後 unload
