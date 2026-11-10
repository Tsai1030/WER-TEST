# library/tts/

TTS client：連到另外啟動的 TTS 服務（services/pocket_tts_server：Pocket TTS，串流；services/tts_server：Supertonic 3），
送文字、收回波形與合成時間。TTS 模型不在專案環境裡：Pocket TTS 需要 numpy 2、Supertonic 在 Jetson GPU 上要用
Jetson AI Lab 的 onnxruntime-gpu（只有 Python 3.12），所以服務各有自己的環境，見各服務的 README。

| 檔案 | 做什麼 |
|---|---|
| client.py | TTSClient：ping()、synthesize(text, voice, lang) → TTSResult（wav、sample_rate、synth_s、audio_s、first_chunk_s）；synthesize_stream(text, voice, lang) 逐段 yield 波形，讀完後結果在 last_result |

## 範例
    with TTSClient("127.0.0.1", 8091) as tts:
        r = tts.synthesize("Hello, how can I help you?", voice="F1", lang="en")
        print(r.synth_s, r.audio_s, r.sample_rate)

    with TTSClient("127.0.0.1", 8092) as tts:          # 串流（ping() 回 "stream": true 的服務）
        for chunk in tts.synthesize_stream("Hello, how can I help you?", voice="alba"):
            ...                                          # 每段到達就可以播放
        print(tts.last_result.first_chunk_s)

## 協定
TCP，每行一個 JSON（UTF-8）。請求 {"text", "voice", "lang"} → 回應 {"ok", "sample_rate", "audio_s", "synth_s", "pcm_f32_b64"}；
音訊為單聲道 float32 PCM 的 base64。失敗回 {"ok": false, "error"}。{"cmd": "ping"} 回服務資訊（串流服務含 "stream": true、"sample_rate"）。
串流：請求加 "stream": true，服務每段回一行 {"ok", "chunk": true, "sample_rate", "pcm_f32_b64"}，最後一行 {"ok", "done": true, "sample_rate", "audio_s", "synth_s", "first_chunk_s"}。
