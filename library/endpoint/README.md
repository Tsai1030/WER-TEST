# library/endpoint/

講完判斷（endpointing）：喚醒後開始收音，以 Silero 串流 VAD 判斷使用者何時說完。

| 檔案 | 做什麼 |
|---|---|
| endpointer.py | EndpointState：逐幀語音機率 → 判斷（不含模型，可直接測試）；decide()：整段機率套用一組參數；Endpointer：Silero 逐幀機率（probs() 離線、stream() 即時）；EndpointResult |

## 判斷規則（時間皆相對於開始收音）
- 連續 min_speech_ms 的機率 ≥ threshold → 開始說話。
- 說話後機率 < threshold − neg_threshold_gap 持續 min_silence_ms → 講完（reason = silence）。
- no_speech_timeout_s 內沒開始說話 → no_speech；說話超過 max_utterance_s → max_length。
- 收音範圍為 [0, end_sec)，不切除句中片段。

## 範例
    ep = Endpointer(); ep.load()
    r = decide(ep.probs(wav), {"min_silence_ms": 800})     # 模擬：機率算一次，可套用多組參數
    s = ep.stream({"min_silence_ms": 800}); r = s.push(chunk)  # 即時

## 參數（預設值）
threshold 0.5、neg_threshold_gap 0.15、min_speech_ms 64、min_silence_ms 800、no_speech_timeout_s 5.0、max_utterance_s 15.0。
Silero 在 16 kHz 時每幀 512 樣本（32 ms），只支援 16 kHz。
