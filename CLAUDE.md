# CLAUDE.md

本專案的完整規格在 `docs/plan.md`。以下規則**不可違反**；若任務需要違反，先停下來詢問。

## 環境

- Windows 原生（PowerShell）、**CPU only**（Intel i3-10110U，無 NVIDIA 顯卡）；之後部署到 Jetson Orin Nano（Linux）。
- 專案路徑 `D:\stt test` **含空格**：命令列中的路徑一律加引號。
- 環境由 **uv** 管理：一律以 `uv run` 執行（例：`uv run python scripts\check_env.py`、`uv run pytest -q`）；
  新增套件用 `uv add`，**不可直接 `pip install`**。PyTorch 必須維持 CPU index（見 `pyproject.toml`）。
- `uv.lock` 必須進版控。
- Parakeet（`parakeet-tdt-v2`）暫緩：只保留空殼與 `enabled: false` 的設定檔，不安裝 NeMo。

## 資料（data/）

- **不可上傳、散佈或公開 `data/` 下任何檔案**（Common Voice 條款禁止重新散佈與辨識說話者）。
  音檔不可放到任何公開位置，結果網頁只放轉寫文字與分數。
- **不可刪除、移動或覆寫 `data/raw/` 下的任何檔案**（archives、female、male）。
- 輸出一律使用匿名 `speaker_id`（`spk01`…），不可輸出 Common Voice 的 `client_id`。

## 命名規則（plan.md 6.5，網頁依此自動分組，不可更改）

| 欄位 | 允許值 |
|---|---|
| model | `granite-470m`、`whisper-small`、`qwen3-asr-0.6b`（`parakeet-tdt-v2` 保留名稱，暫不使用） |
| condition | `clean`、`snr20`、`snr10`、`snr05`、`snr00`、`snrm05`、`snrm10`、`reverb`、`lowvol`、`noise_only`（`snrm` = 負 SNR，Round 2 起） |
| method | `baseline`、`vad`、`denoise`、`loudnorm`、`beam`、`greedy`；組合以 `+` 連接，如 `vad+denoise`（`beam` / `greedy` 為解碼方法，Round 3 起） |

## 分層原則（plan.md 4.1）

| 層 | 職責 | 可以依賴 |
|---|---|---|
| `scripts/` | 解析命令列參數、呼叫 library（不放邏輯） | `library` |
| `library/runner/` | 組合流程 | 其他所有 library 子模組 |
| `library/{data, degrade, methods, models, textnorm, scoring, monitor, export}` | 單一職責 | 只能依賴 `library/utils/` |
| `library/utils/` | 共用工具 | 無 |

子模組之間**不互相 import**（例如 `scoring` 不能 import `models`），只有 `runner` 負責串接。

## 撰寫規範（plan.md 第 14 章）

- Python 3.11、全面使用型別標註；每個公開函式要有 docstring（用途、參數、回傳值、例外）。
- 路徑一律透過 `library/utils/paths.py` 並使用 `pathlib.Path`；不寫死絕對路徑、不手動拼接 `\` 或 `/`。
- 讀寫文字檔（CSV / JSON / TSV / YAML）一律指定 `encoding="utf-8"`（Windows 預設 cp950）。
- 程式中不呼叫 shell 指令（`ls`、`cp`…）；改用 Python 標準函式庫（`shutil`、`tarfile`）。
- 所有 `scripts/` 都要有 `if __name__ == "__main__":` 保護（multiprocessing 在 Windows 也需要）。
- **不可假設有 GPU**：裝置一律由設定檔 `device` 決定；`cuda` 不可用時自動退回 `cpu` 並記錄警告。
  GPU 記憶體量測只在偵測到 CUDA 時進行，否則 `peak_gpu_mb = None`。
- 參數一律來自 `configs/` 下的設定檔；程式中只放預設值。
- 亂數一律透過 `library/utils/seed.py` 並記錄 seed。
- 重量級套件（torch、transformers、模型套件）在 `load()` 內才 import，讓 `import library` 保持輕量。
- 新增模型 / 方法：繼承 base 類別 → 在 registry 註冊 → 新增設定檔 → 更新該資料夾 README。
- 每個子模組至少一個單元測試；`test_smoke_pipeline.py` 必須在 1 分鐘內跑完。
- 以 ruff 格式化與檢查（`uv run ruff check .`、`uv run ruff format .`）。
- Commit 訊息格式：`[模組] 動作：說明`，例如 `[scoring] add：配對 bootstrap`。
- 所有檔案以 UTF-8 儲存。
