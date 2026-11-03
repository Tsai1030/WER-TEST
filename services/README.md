# services/

外部服務：各有自己的環境或程式，不 import library，也不裝進專案環境；library/llm、library/tts 的 client 透過網路連線。

| 資料夾 | 服務 | 預設位址 |
|---|---|---|
| tts_server/ | TTS：Supertonic 3（Python 3.12 的 uv 環境；板子 GPU 用 Jetson AI Lab 的 onnxruntime-gpu） | 127.0.0.1:8091 |
| llama_cpp/ | LLM：llama.cpp 的 llama-server（自行編譯，OpenAI 相容 API） | http://127.0.0.1:8090 |

## 整段語音助理實測（板子）
1. 啟動 LLM：見 llama_cpp/README.md（建議 Qwen3-1.7B Q4_K_M）。
2. 啟動 TTS：`cd services/tts_server && uv run python server.py`（看到 TTS_READY）。
3. 接上 USB 麥克風與喇叭（需要系統的 libportaudio2），在專案根目錄執行：
   `uv run python scripts/10_assistant_live.py --list-devices`，再以 `--input-device` / `--output-device` 選裝置執行。
