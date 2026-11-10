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
| vad_sweep.py | run_vad_sweep()：單一模型 × 多組 VAD 設定逐句辨識，寫 sweep.csv（可續跑）；summarize_sweep()：各設定 WER、刪除數、男女聲、噪音誤放行與選擇規則 |
| wake.py | Round 4 自動測試：load_wake_config()；run_kws_negative()（誤喚醒）、run_endpoint_sim()（講完判斷模擬）、run_kws_positive()（自錄音檔）、run_pipeline()（整段流程：講完判斷 → 辨識）、run_standby()（待機成本，各元件在獨立子行程中量）；summarize_kws()、summarize_endpoint()（含推薦規則）、summarize_pipeline()（與 oracle 比較） |
| live.py | Round 4 麥克風：run_recorder()（07_record_wake.py 引導錄音）、LiveDemo（09_live_demo.py 即時實測：喚醒 → 回覆 → 收音 → 講完判斷 → 辨識） |
| assistant.py | 整段語音助理手動實測（10_assistant_live.py）：load_assistant_config()；LiveAssistant（麥克風 → 講完判斷 → STT → LLM 串流 → 一句一句 TTS 並播放，TTS 服務支援串流時邊合成邊播；第一段可在逗號先切；印出並記錄各段時間）；split_sentences()、turn_timings()、compute_breakdown()（VAD + STT + LLM + TTS 純計算時間合計）；device_rate()（音效卡不支援 16 kHz 時改用裝置取樣率，收音即時轉成 16 kHz、播放前轉成裝置取樣率） |
| report.py | 05_quick_report.py 的計算：整體 WER、ΔWER 與 CI、幻覺率、正規化影響、男女聲、資源、一句話結論（定義同 dashboard/sections.md） |

## 流程
載入模型 → 每個方法重設資源統計 → 每個條件讀取劣化音檔 → 方法處理 → 模型辨識
→ 正規化 → 對齊 → 寫入一列 → 該模型結束後 unload
