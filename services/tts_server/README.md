# services/tts_server/

TTS 服務：Supertonic 3（約 400 MB，英文、日文等，不支援中文），給 library/tts 的 TTSClient 連線。
這是外部服務，有自己的 uv 環境（本資料夾的 pyproject.toml / uv.lock），不 import library，也不裝進專案環境。

## 為什麼分開
- 板子 GPU 要用 NVIDIA Jetson AI Lab 的 onnxruntime-gpu 1.24.0（PyPI 的 aarch64 版沒有 Orin sm_87 的 GPU 程式碼，會出現
  `cudaErrorNoKernelImageForDevice`），而它只有 Python 3.12 版；專案環境固定 Python 3.11。
- 開發機（Windows）同一份設定會裝 CPU 版 onnxruntime。

## 啟動（在本資料夾）
    uv run python server.py                          # 板子：GPU、補齊 96 / 128，埠號 8091
    uv run python server.py --device cpu --pad none  # 開發機：CPU
第一次會自動下載模型（約 400 MB）到 `models/supertonic-3/`（不進版控）；載入後先合成一句暖機，看到 `TTS_READY` 就可以連線。
預設只接受本機連線（--host 127.0.0.1）。

## 補齊（--pad）
GPU 上每次輸入長度改變，cuDNN 卷積要重新準備（四個模型合計約 0.6 秒）。把文字補到 96 格、latent 補到 128 格（約 8.9 秒音訊）後
每句約 0.33 秒；代價是輸出和原版不完全相同（時長差約 1%）。超過長度的句子照原樣合成。CPU 上沒有這個問題，用 --pad none。
詳見 docs/experiment_log.md 的 TTS Round 1。

## 參數
--host、--port（8091）、--device（cuda / cpu）、--pad（96,128 / none）、--steps（8）、--threads（4）、--model-dir（預設 models/supertonic-3）。
