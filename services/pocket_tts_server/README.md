# services/pocket_tts_server/

串流 TTS 服務：Kyutai Pocket TTS（100M，英文；另有法、德、西、葡、義、荷語模型），給 library/tts 的 TTSClient 連線。
邊合成邊回傳音訊（約每 80 ms 一段），LLM 交出一句後很快就能開始播放；整段語音助理（configs/assistant.yaml）預設用這個服務。
這是外部服務，有自己的 uv 環境（本資料夾的 pyproject.toml / uv.lock），不 import library，也不裝進專案環境。

## 為什麼分開
- pocket-tts 需要 numpy 2，專案環境是 numpy 1.26。
- PyTorch 來源和專案相同：板子（Linux aarch64）用官方 cu132 版、其他平台用 CPU 版。

## 啟動（在本資料夾）
    uv run python server.py                  # 板子：GPU，埠號 8092，聲音 alba
    uv run python server.py --device cpu     # 開發機：CPU
第一次會自動下載模型（約 240 MB）到 Hugging Face 快取；載入後先合成兩句暖機，看到 `TTS_READY` 就可以連線。
預設只接受本機連線（--host 127.0.0.1）。

## 為什麼用 GPU（TTS Round 2）
板子 CPU 太弱：100M 模型在 CPU 上 RTF 1.66（4 執行緒）～2.82（1 執行緒），合成比播放慢；GPU 上第一段約 0.12 秒、RTF 約 0.6、
不會斷音。18M 小模型（--language english_18m_2026-10）在 CPU 單執行緒剛好跟得上播放（RTF 約 1.0），沒有餘裕。
板子記憶體比 Supertonic 多約 0.5 GB（多半是 PyTorch 與 CUDA 函式庫本身）。詳見 docs/experiment_log.md 的 TTS Round 2。

## 協定
同 library/tts/client.py：請求加 `"stream": true` 時每產生一段回一行 `{"ok": true, "chunk": true, ...}`，最後一行
`{"ok": true, "done": true, "first_chunk_s", "synth_s", "audio_s", ...}`；不加時一次回傳整句（與 Supertonic 服務相同）。
`{"cmd": "ping"}` 回傳的 `"stream": true` 讓 client 知道可以串流，`"sample_rate"` 為 24000。

## 參數
--host、--port（8092）、--device（cuda / cpu）、--language（預設 english；例如 english_18m_2026-10、french）、
--voice（預設 alba；內建聲音名稱或 wav 檔，用於聲音複製）、--threads（PyTorch CPU 執行緒數，預設 1）。
內建聲音與授權見 https://huggingface.co/kyutai/tts-voices 。
