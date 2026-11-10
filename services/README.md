# services/

外部服務：各有自己的環境或程式，不 import library，也不裝進專案環境；library/llm、library/tts 的 client 透過網路連線。

| 資料夾 | 服務 | 預設位址 |
|---|---|---|
| pocket_tts_server/ | TTS（預設）：Pocket TTS 100M，串流（Python 3.11 的 uv 環境，PyTorch 同專案來源） | 127.0.0.1:8092 |
| tts_server/ | TTS：Supertonic 3，整句合成（Python 3.12 的 uv 環境；板子 GPU 用 Jetson AI Lab 的 onnxruntime-gpu） | 127.0.0.1:8091 |
| llama_cpp/ | LLM：llama.cpp 的 llama-server（自行編譯，OpenAI 相容 API） | http://127.0.0.1:8090 |

## 整段語音助理實測（板子）
1. 啟動 LLM：見 llama_cpp/README.md（建議 Qwen3-1.7B Q4_K_M）。
2. 啟動 TTS：`cd services/pocket_tts_server && uv run python server.py`（看到 TTS_READY）。改用 Supertonic 時啟動 tts_server，
   並在步驟 3 加 `--tts-port 8091 --voice F1`。
   要最快時板子先 `sudo nvpmodel -m 2`（MAXN_SUPER）與 `sudo jetson_clocks`（鎖定最高時脈，重開機後恢復）；耗電與發熱較高。
3. 接上 USB 麥克風與喇叭（需要系統的 libportaudio2），在專案根目錄執行：
   `uv run python scripts/10_assistant_live.py --list-devices`，再以 `--input-device` / `--output-device` 選裝置執行（USB 音效卡只支援 44.1 / 48 kHz 也可以直接用，程式會自動轉換取樣率）。
