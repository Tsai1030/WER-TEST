# 資料契約（Data Contract）

結果網頁讀取 `results/<round>/` 下的三個檔案：`results.csv`、`resources.csv`、`meta.json`。
所有彙總（整體 WER、信賴區間、變好/變差、對齊）都由網頁從逐句資料自動計算；
**實驗程式只需輸出逐句原始結果。**

本文件是 `library/export/schema.py` 與 `library/export/validate.py` 的依據，兩者必須保持一致。
來源：`docs/plan.md` 第 6 章、第 10.3、10.5 節。

## 通用規則

- 編碼一律 **UTF-8**；CSV 以逗號分隔、含標題列，文字欄位依 CSV 規則加引號跳脫。
- 空值（「留空」）寫成空欄位，不寫 `None`、`nan`、`null`。
- 浮點數使用 `.` 為小數點；不加千分位。
- 檔案中**不放音檔、不放音檔路徑以外的說話者資訊**；`speaker_id` 一律為匿名 ID（`spk01`…），
  不可出現 Common Voice 的 `client_id`。

---

## 1. `results.csv`（每列一個「模型 × 條件 × 方法 × 句子」）

| 欄位 | 型別 | 說明 |
|---|---|---|
| model | str | 見「命名規則」 |
| condition | str | 見「命名規則」 |
| method | str | 見「命名規則」 |
| utt_id | str | `utt001`…`utt050`；純噪音為 `noise01`…`noise10` |
| speaker_id | str | 匿名 ID；純噪音留空 |
| gender | str | `female` / `male`；純噪音留空 |
| audio_sec | float | 原始音檔秒數（計算 RTF 用） |
| ref_raw | str | 純噪音為空 |
| hyp_raw | str | 模型原始輸出 |
| ref_norm | str | 正規化後 |
| hyp_norm | str | 正規化後 |
| wer | float | 0–1（插入多時可 > 1）；純噪音為空 |
| S | int | 替換數；純噪音為 0 |
| D | int | 刪除數；純噪音為 0 |
| I | int | 插入數；純噪音為 hyp 詞數 |
| N | int | reference 詞數；純噪音為 0 |
| latency_ms | float | 方法處理 + 模型辨識的總時間 |
| method_ms | float | 僅方法處理時間 |
| detected_lang | str | 多語模型偵測語言，其他留空 |

**一致性檢查（validate）**

- `wer = (S + D + I) / N`（N > 0 時），誤差 < 1e-6。
- `condition = noise_only` 時：`S = D = 0`、`N = 0`、`wer` 為空、`ref_raw` 為空。
- `method_ms ≤ latency_ms`。
- VAD 判斷無語音時不呼叫模型：`hyp_raw` 為空字串，該列仍需寫出。
- 列數 = 模型數 × 方法數 ×（語音條件數 × 語音句數 + 純噪音片段數）。
  Round 1：3 × 4 ×（7 × 50 + 10）= 4,320 列。
- 「模型 × 條件 × 方法 × utt_id」不可重複。

## 2. `resources.csv`（每列一個「模型 × 方法」）

| 欄位 | 型別 | 說明 |
|---|---|---|
| model | str | |
| method | str | |
| peak_gpu_mb | float | 該組合執行期間 GPU 峰值；CPU only 時留空 |
| peak_ram_mb | float | 該組合執行期間 RAM 峰值（本輪主要記憶體指標） |
| model_size_mb | float | 模型權重檔大小 |
| load_time_s | float | 模型載入時間 |
| gpu_measure | str | `none`（CPU only）、`torch` 或 `pynvml`（不同測法不可直接比較） |

- `gpu_measure = none` 時 `peak_gpu_mb` 必須為空。
- Round 1 列數：3 × 4 = 12 列。

## 3. `meta.json`

```json
{
  "round": "round1",
  "date": "2026-11-xx",
  "git_commit": "abc1234",
  "device": "cpu",
  "cpu": "Intel Core i3-10110U (2C/4T)",
  "ram_gb": 15.8,
  "gpu": "none",
  "num_threads": 4,
  "env": "Windows 11（原生）",
  "python": "3.11.x",
  "precision": "fp32",
  "dataset": "Common Voice Scripted Speech 26.0 US English F+M test, 25+25, seed 42, + 10 noise-only",
  "normalizer": "whisper EnglishTextNormalizer + filler removal",
  "nano_budget_mb": 8192,
  "packages": { "torch": "x.y", "transformers": "x.y", "qwen-asr": "x.y" },
  "models": {
    "whisper-small": { "hf_id": "openai/whisper-small", "framework": "transformers", "decoding": "beam 5, temperature 0" }
  },
  "methods": { "vad": { "threshold": 0.5, "min_silence_ms": 300, "pad_ms": 200 } },
  "conditions": { "snr10": { "snr_db": 10, "noise_source": "DEMAND (16k)" } }
}
```

- `models` / `methods` / `conditions` 的 key 必須符合命名規則。
- 另含 `seed`、`resumed`（是否經過續跑）與 `matrix`：
  `{"models": [...], "methods": [...], "conditions": [...], "n_speech": 50, "n_noise": 10}`，用於驗證 results.csv 列數。
- `detected_lang` 只記錄自動偵測的語言；設定檔強制語言時留空（Round 1 全部留空）。
- 若縮減規模、改用 int8 或降噪改用 noisereduce，必須寫入 meta.json（例如在對應 model / method 下記錄實際實作）。

## 4. 命名規則（網頁依此自動分組，不可隨意更改）

| 欄位 | 允許值 |
|---|---|
| model | `granite-470m`、`whisper-small`、`qwen3-asr-0.6b`（`parakeet-tdt-v2` 保留名稱，暫不使用） |
| condition | `clean`、`snr20`、`snr10`、`snr05`、`snr00`、`reverb`、`lowvol`、`noise_only` |
| method | `baseline`、`vad`、`denoise`、`loudnorm`；組合以 `+` 連接，如 `vad+denoise` |

- 組合方法的順序即處理順序（`vad+denoise` 為先 VAD 再降噪）。
- 每個 model × condition 必須有 `baseline`，否則無法計算 ΔWER。

## 5. 附：`data/clean/manifest.csv`（不交給網頁，僅供追溯）

| 欄位 | 型別 | 說明 |
|---|---|---|
| utt_id | str | `utt001`…`utt050`；純噪音為 `noise01`…`noise10` |
| wav_path | str | 相對於專案根目錄的路徑 |
| ref_raw | str | Common Voice 原始句子 |
| speaker_id | str | 匿名 ID（`spk01`…），純噪音留空 |
| gender | str | `female` / `male`（由 female_feminine / male_masculine 轉換），純噪音留空 |
| age | str | Common Voice 年齡層（例：thirties），可留空 |
| audio_sec | float | 秒數 |
| source_file | str | 原始 MP3 檔名（追溯用，不對外公開） |

## 6. 隱私與散佈（plan.md 10.5）

- 網頁只放轉寫文字與分數，**不放音檔**（Common Voice 條款）。
- `speaker_id` 使用匿名 ID，不使用 Common Voice 的 client_id。
- 網頁預設為私人，分享前需確認內容。
