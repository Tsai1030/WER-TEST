# library/tts/

TTS client：連到另外啟動的 TTS 服務（services/tts_server，Supertonic 3），送文字、收回波形與合成時間。
TTS 模型不在專案環境裡：Supertonic 在 Jetson GPU 上要用 Jetson AI Lab 的 onnxruntime-gpu（只有 Python 3.12），
所以服務有自己的環境，見 services/tts_server/README.md。

| 檔案 | 做什麼 |
|---|---|
| client.py | TTSClient：ping()、synthesize(text, voice, lang) → TTSResult（wav、sample_rate、synth_s、audio_s） |

## 範例
    with TTSClient("127.0.0.1", 8091) as tts:
        r = tts.synthesize("Hello, how can I help you?", voice="F1", lang="en")
        print(r.synth_s, r.audio_s, r.sample_rate)

## 協定
TCP，每行一個 JSON（UTF-8）。請求 {"text", "voice", "lang"} → 回應 {"ok", "sample_rate", "audio_s", "synth_s", "pcm_f32_b64"}；
音訊為單聲道 float32 PCM 的 base64。失敗回 {"ok": false, "error"}。{"cmd": "ping"} 回服務資訊。
