# 決策紀錄

重要決策與理由（模型、資料集、指標的選擇）。初始決策見 `plan.md` 第 2 章「已確定的決策摘要」；
之後的變更依下列格式由新到舊附加。

## 格式

```markdown
## YYYY-MM-DD：<決策標題>
- 決定：
- 理由：
- 替代方案：
- 影響範圍（設定檔 / 模組 / 報告）：
```

---

## 2026-10-30：Whisper 改用 small，保留完整矩陣；Jetson 部署預計 q4
- 決定：模型名稱 `whisper-turbo` 改為 `whisper-small`（hf_id `openai/whisper-small`，transformers fp32、beam 5、temperature 0）；
  Round 1 保留完整 8 條件 × 4 方法 × 50 句；`num_threads` 維持 4。plan.md 6.5 命名規則同步修改。
- 理由（M0 實測，fp32、5 句）：
  - large-v3-turbo 24.07 s/句，完整矩陣合計 12.2 小時；Whisper 佔 9.6 小時，是唯一瓶頸
    （Whisper 每句都補到 30 秒跑 32 層 encoder，與句長無關）。
  - faster-whisper int8 反而較慢（28.23 s/句）：i3-10110U 無 AVX-512 VNNI，int8 無法加速。
  - 2 緒時 turbo 慢 65%（39.72 s/句），granite 持平，qwen 快 13%，整體不划算。
  - small 11.57 s/句，合計約 7.2 小時，一晚可跑完，且保留 snr20（驗證「高 SNR 降噪可能變差」的關鍵條件）。
  - 本研究看方法開/關的 ΔWER 與架構類別，Whisper 在此代表 encoder-decoder 架構，模型大小不是重點。
- 部署：Jetson 預計使用 q4。開發機上不測 q4（int8 已不加速；三個模型的 q4 需各自不同推論引擎，
  反而多出框架差異的變因）。改在 Jetson 以 q4 部署版本複驗 baseline 與最佳方法（plan.md 11.5）。
- 替代方案：照 8.5 縮減條件並保留 turbo（約 8.8 小時，但失去 snr20、snr05）；完整矩陣保留 turbo 分兩晚（約 12.2 小時）。
- 限制：若 Jetson 最後部署 turbo，small 上量到的改善幅度不一定能直接套用，需靠 Jetson 複驗確認。
- 影響範圍：`configs/models/whisper-small.yaml`（由 whisper-turbo.yaml 改名）、`configs/round1.yaml`、
  `library/models/whisper.py`、`library/export/schema.py`、CLAUDE.md、dashboard/data_contract.md、plan.md。

## 2026-10-30：Qwen3-ASR 改用 transformers 原生版，不用 qwen-asr 套件
- 決定：`qwen3-asr-0.6b` 的 hf_id 改為 `Qwen/Qwen3-ASR-0.6B-hf`，framework 改為 `transformers`；
  `models` extra 移除 `qwen-asr`；主相依改為 `transformers>=5.16`（目前鎖 5.22.0）。
- 理由：`qwen-asr` 0.0.6 鎖定 `transformers==4.57.6`，但 Granite Speech 5.0（`GraniteSpeech5ForCTC`）
  需要 transformers ≥ 5.16，兩者無法裝在同一環境。transformers 5.13 起已內建 `Qwen3ASRForConditionalGeneration`，
  可直接載入官方 `-hf` 權重，且少裝 gradio、flask 等不需要的套件。
- 替代方案：分開兩個環境（增加維護成本）；改用 SenseVoice-Small（換了模型，失去 Audio-LLM 架構）。
- 影響範圍：`configs/models/qwen3-asr-0.6b.yaml`、`library/models/qwen3_asr.py`、`pyproject.toml`、`uv.lock`；
  meta.json 中 qwen3-asr 的 framework 記為 transformers。

## 2026-10-30：降噪改用 noisereduce
- 決定：`configs/methods/denoise.yaml` 的 impl 改為 `noisereduce`。
- 理由：`deepfilternet` 0.5.6 可安裝但 import 失敗（`No module named 'torchaudio.backend'`，
  torchaudio 2.1 起已移除該模組），且該套件已長期未更新。依 plan.md 8.3 的備案處理。
- 替代方案：降版 torchaudio（會與 torch 2.15 不相容）；自行修補 DeepFilterNet 原始碼。
- 影響範圍：Exp2 降噪的結果代表 noisereduce（頻譜閘控），報告與 meta.json 需註明。

## 2026-10-30：模型權重鎖定 revision
- 決定：三個模型設定檔加上 `revision`（Hugging Face commit hash），載入時固定使用該版本，並寫入 meta.json。
- 理由：M0 首次載入 Granite 時，transformers 偵測到上游已更新（947f59a → 1dd23f4）而自動下載新版，
  代表不鎖版本時結果可能在不知情下改變。
- 影響範圍：`configs/models/*.yaml`、`library/models/*.py`（from_pretrained 傳入 revision）。

## 2026-10-30：模型快取放 D 槽
- 決定：使用者層級環境變數 `HF_HOME=D:\stt test\.hf_cache`（已在 .gitignore）。
  Granite 與 Qwen3-ASR 已存在於 C 槽舊快取，直接複製到 D 槽（C 槽原檔未動），只下載 Whisper。
- 影響範圍：新開的 PowerShell 才會讀到此環境變數。

## 2026-10-30：Parakeet TDT 0.6B v2 暫緩
- 決定：只保留 `library/models/parakeet.py` 空殼與 `configs/models/parakeet-tdt-v2.yaml`（enabled: false）。
- 理由：依賴 NeMo，官方以 Linux 為主，Windows 原生安裝困難。
- 替代方案：WSL2 或 Jetson 上補跑同一份設定。
- 影響範圍：本輪缺少 Transducer 架構比較，報告需註明。
