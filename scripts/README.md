# scripts/

命令列入口，只解析參數並呼叫 library，不放邏輯。依編號順序執行。

| 檔案 | 做什麼 | 主要參數 |
|---|---|---|
| check_env.py | 檢查 Python、套件版本、可用裝置、三個模型能否載入，並實測每句秒數 | --models, --bench 5 |
| 01_prepare_data.py | 抽樣 50 句、轉檔、產生 manifest | --config |
| 02_make_degraded.py | 產生所有劣化條件音檔 | --config |
| 03_run_matrix.py | 執行實驗矩陣（可續跑） | --config, --no-resume, --save-processed |
| 04_validate_results.py | 驗證三個輸出檔 | results_dir |
| 05_quick_report.py | 終端機印出各模型 × 方法整體 WER 摘要 | results_dir |
