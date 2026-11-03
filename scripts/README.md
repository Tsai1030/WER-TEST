# scripts/

命令列入口，只解析參數並呼叫 library，不放邏輯。依編號順序執行。

| 檔案 | 做什麼 | 主要參數 |
|---|---|---|
| check_env.py | 檢查 Python、套件版本、可用裝置、三個模型能否載入，並實測每句秒數 | --models, --bench 5 |
| 01_prepare_data.py | 抽樣 50 句、轉檔、產生 manifest（設定 `sampling.exclude_manifest` 時排除該 manifest 的說話者） | --config |
| 02_make_degraded.py | 產生所有劣化條件音檔 | --config |
| 03_run_matrix.py | 執行實驗矩陣（可續跑） | --config, --no-resume, --save-processed |
| 04_validate_results.py | 驗證三個輸出檔 | results_dir |
| 05_quick_report.py | 終端機印出各模型 × 方法整體 WER 摘要 | results_dir |
| 06_vad_sweep.py | VAD 參數掃描（可續跑）並印出各設定摘要與依規則選出的設定 | --config, --summary-only |
| 07_record_wake.py | Round 4：引導錄製「小咪小咪」、音近詞與回覆聲到 data/wake/（已錄過的跳過） | --device, --list-devices, --redo |
| 08_wake_eval.py | Round 4 自動測試：誤喚醒、講完判斷模擬、自錄音檔的正確喚醒率；另可指定 standby（待機成本）、pipeline（整段流程），印出彙總與推薦設定 | --config, --parts, --summary-only |
| 09_live_demo.py | Round 4 手動實測：麥克風即時跑喚醒 → 回覆 → 收音 → 講完判斷 → 辨識，顯示各段時間 | --kws-only, --threshold, --min-silence-ms, --no-listen-during-reply |
| 10_assistant_live.py | 整段語音助理手動實測：麥克風說英文 → 講完判斷 → STT → LLM → TTS → 喇叭，每輪印出各段時間（講完 / STT 開始 → 開始出聲、全部合成完、全部播完）；要先啟動 services/ 的 LLM 與 TTS 服務 | --list-devices, --input-device, --output-device, --asr-device, --min-silence-ms, --history-turns, --no-save |
