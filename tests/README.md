# tests/

    pytest -q

| 檔案 | 測什麼 |
|---|---|
| test_align.py | S/D/I 計算、空字串、全錯等邊界情況 |
| test_wer.py | corpus_wer 為加權而非平均；與 jiwer 一致 |
| test_stats.py | 配對 bootstrap 可重現、顯著 / 不顯著判定、變好 / 變差 / 不變統計 |
| test_textnorm.py | 標點、縮寫、數字、贅詞 |
| test_degrade.py | 實測 SNR 與目標誤差、同 seed 結果相同 |
| test_manifest.py | 欄位、檔案存在、取樣率 |
| test_smoke_pipeline.py | 2 句 × granite × 全部方法 × clean / noise_only 跑完整流程（開發機約 1 分鐘）；`SMOKE_ALL=1` 時跑全部模型 × 方法 × 條件（M4 驗收，約 20 分鐘） |
| test_runner.py | 設定讀取與命名驗證、矩陣展開順序、完整執行通過驗證、中斷續跑不重複、已完成模型不重載、--no-resume 備份 |
| test_export.py | ResultWriter 附加與空欄位、validate_results 能抓出各類規格錯誤 |
| test_report.py | 摘要計算：整體 WER 加權、ΔWER 顯著判定、幻覺率、假錯誤、一句話結論 |
| test_monitor.py | ResourceMonitor 峰值 RAM、CPU only 時 GPU 為 None、Timer |
| test_methods.py | 方法註冊表、組合、baseline、loudnorm 達標、降噪基本行為、VAD 無語音回傳空陣列 |
| test_models.py | 模型註冊表符合命名規則、parakeet 停用、未 load 時報錯、cuda 退回 cpu、矩陣規模 |
