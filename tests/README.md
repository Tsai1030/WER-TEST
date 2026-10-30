# tests/

    pytest -q

| 檔案 | 測什麼 |
|---|---|
| test_align.py | S/D/I 計算、空字串、全錯等邊界情況 |
| test_wer.py | corpus_wer 為加權而非平均；與 jiwer 一致 |
| test_textnorm.py | 標點、縮寫、數字、贅詞 |
| test_degrade.py | 實測 SNR 與目標誤差、同 seed 結果相同 |
| test_manifest.py | 欄位、檔案存在、取樣率 |
| test_smoke_pipeline.py | 2 句 × 1 模型跑完整流程（1 分鐘內） |
