# 實驗紀錄

每次實驗依下列格式由新到舊附加。

## 格式

```markdown
## YYYY-MM-DD：<實驗名稱>
- 設定：configs/<檔名>.yaml（git commit：）
- 範圍：模型 / 方法 / 條件 / 句數
- 觀察：
- 結論：
- 下一步：
```

---

## 2026-11-04：Pipeline Round 2（換成 Pocket TTS 串流的整段流程；25W / MAXN_SUPER；第一段逗號先切）
- 設定：configs/assistant.yaml（Pocket TTS、port 8092、聲音 alba）、library/runner/assistant.py、services/pocket_tts_server（git commit：待提交）
- 範圍：板子上用 Pipeline Round 1 的英文問句音檔代替麥克風，跑 LiveAssistant 的每一輪（暫存腳本）：講完判定的等待固定模擬 1.0 秒，
  播放改成依音訊長度等待（沒有接喇叭）；第 1 輪暖機不計入。granite-470m fp16 GPU → Qwen3-1.7B Q4_K_M（llama-server）→ Pocket TTS 100M GPU 串流
- 電源：25W（時脈未鎖定）與 MAXN_SUPER（`sudo nvpmodel -m 2` + `sudo jetson_clocks`：CPU 1,728 MHz、GPU 1,020 MHz、記憶體 3,199 MHz）

### 觀察（講完 → 開始出聲，中位數；含模擬的 1.0 秒等待靜音）
| 設定 | 輪數 | 開始出聲（範圍） | STT | LLM 第一段 | TTS 第一段 | LLM token/s | 整板記憶體峰值 |
|---|---|---|---|---|---|---|---|
| Supertonic · 25W · 整句（Pipeline Round 1 整合測試） | 3 | 1.83 s（1.70–1.99） | 0.13 | 0.31 | 0.36 | 28.9 | 6,413 MB（Pipeline Round 1） |
| Pocket · 25W · 整句 | 7 | 1.67 s（1.45–1.92） | 0.11 | 0.44 | 0.12 | 28.6 | 6,782 MB |
| Pocket · MAXN_SUPER · 整句 | 9 | 1.49 s（1.31–1.67） | 0.06 | 0.36 | 0.08 | 35.7 | 6,565 MB |
| **Pocket · MAXN_SUPER · 第一句逗號先切** | 9 | **1.42 s（1.32–1.67）** | 0.06 | 0.28 | 0.08 | 35.8 | 6,591 MB |

- 所有設定播放都沒有中斷（全部播完 − 開始出聲 = 回覆音訊長度）。同樣 3 題：Supertonic 版 1.83 s，建議設定 1.40 s。
- MAXN_SUPER + jetson_clocks：扣掉等待靜音的計算部分 0.67 → 0.49 s（−27%），STT 0.11 → 0.06 s；溫度約 50°C。
  25W 時麥克風實測的 STT 偶爾 0.4–1.2 s，推測是閒置降頻（GPU 306 MHz、記憶體 204 MHz）後升頻要時間；鎖定時脈後沒有再出現。
- 第一段逗號先切（逗號前至少 3 個字）：只有第一句有逗號的題目會變（10 題中 3 題：1.94 → 1.82、1.67 → 1.37、1.67 → 1.62 s），
  其餘相同。Pipeline Round 1 同樣做法沒效果是因為 Supertonic 每段要 0.3 s 以上；現在 TTS 第一段只要 0.08 s，等 LLM 寫完整句才是瓶頸。
- 講完判定的靜音不縮短：Round 4 的 700 ms 在句中停頓 0.9 秒時截斷率 91%（1,000 ms 為 4.7%），整段流程 WER 顯著變差（+1.77 個百分點）。
- 整板記憶體峰值比 Supertonic 版高約 0.2–0.4 GB（剩約 0.7 GB），板子上不要再同時跑其他大程式。

### 結論
- 預設改為 Pocket TTS 串流 + 第一段逗號先切（configs/assistant.yaml）；板子用 MAXN_SUPER + jetson_clocks 時講完到開始出聲約 1.4 s，
  其中 1.0 s 是等待靜音、LLM 第一段約 0.3 s、STT 與 TTS 各不到 0.1 s。
- 限制：合成語音輸入、每組 8–10 題、各跑一次；沒有接麥克風與喇叭；MAXN_SUPER 長時間運作的溫度與耗電沒量。

### 下一步
- 用麥克風與耳機實測（`--input-device`、`--output-device`）；Pocket TTS 的聽感確認與換聲音。

---

## 2026-11-04：TTS Round 2（串流 TTS：Pocket TTS、Soprano，板子）
- 設定：暫存腳本（Pocket 整理為 services/pocket_tts_server）
- 範圍：TTS Round 1 的英文 10 句助理回覆（每組先 2 句暖機）；Pocket TTS 4.0.1（english 100M、english_18m_2026-10，聲音 alba）、
  Soprano 0.2.0（Soprano-1.1-80M，transformers 後端；lmdeploy 後端板子不能用）；PyTorch 2.15.0+cu132；25W、時脈未鎖定，
  llama-server 與 Supertonic 服務已載入但閒置
- 量測：第一段 = 送出文字到收到第一段音訊；RTF = 合成時間 ÷ 音訊長度；斷音 = 第一段一到就開始播、之後某段到達時前面已播完（> 50 ms）；
  記憶體 = tegrastats 整板峰值 − 啟動前；可懂度 = 開發機 granite-470m 辨識合成語音的 WER

### 觀察
| 設定 | 第一段（中位數） | RTF | 斷音句數 | 板子記憶體增加 | WER |
|---|---|---|---|---|---|
| Supertonic 3 · GPU 補齊（TTS Round 1，整句才出聲） | 0.33 s | 0.09 | — | 948 MB | 1.1% |
| **Pocket 100M · GPU** | **0.12 s** | **0.60** | **0 / 10** | 1,443 MB | **1.1%** |
| Pocket 100M · GPU · LLM 同時不停產生文字 | 0.19 s | 1.04 | 1 / 10（63 ms） | 1,524 MB | 1.1% |
| Pocket 100M · CPU 4 執行緒 | 0.34 s | 1.66 | 10 / 10 | 1,149 MB | 2.2% |
| Pocket 100M · CPU 1 執行緒（套件預設） | 0.58 s | 2.82 | 10 / 10 | 995 MB | 1.1% |
| Pocket 18M · CPU 1 執行緒 | 0.23 s | 0.99 | 0 / 10 | 499 MB | 1.1% |
| Pocket 18M · CPU 4 執行緒 | 0.25 s | 1.09 | 10 / 10 | 640 MB | 1.1% |
| Pocket 18M · GPU | 0.23 s | 0.86 | 0 / 10 | 1,179 MB | 1.1% |
| Soprano 80M · GPU | 0.45 s | 1.41 | 10 / 10 | 1,675 MB | 4.4% |
| Soprano 80M · CPU 4 執行緒（2 句後停止） | 1.19 s | 3.84 | 2 / 2 | 649 MB | — |

- 所有設定唯一共同的辨識錯誤是 three thirty 被寫成 3.30（數字格式）。Soprano 第 1 句（"I'm here, I'm here."）產生 32 秒的幻覺。
- 板子 CPU 太弱：Pocket 100M 在 CPU 上都比播放慢；pocket_tts 在 import 時把 PyTorch 執行緒設為 1，要在載入後再設。
- Pocket 用 GPU 時 PyTorch 實際配置約 486 MB，板子記憶體增加多出來的部分是 PyTorch 與 CUDA 函式庫本身；18M 在 GPU 上反而比 100M 慢
  （CPU 送 GPU 指令的負擔為主）。
- Soprano 與 Qwen3-TTS 一樣要用 transformers 逐 token 產生，速度被板子 CPU 拖住（同 Pipeline Round 1 的 LLM）；Qwen3-TTS（0.6B）未測。
- Pocket 不能裝進專案環境（需要 numpy 2，專案是 1.26），改為外部服務（services/pocket_tts_server，協定加上串流）。

### 結論
- TTS 改用 Pocket TTS 100M（GPU、串流）：第一段約 0.12 s，比 Supertonic 快約 2.7 倍；記憶體多約 0.5 GB。
- 限制：只測英文 10 句、預設聲音；音質只量可懂度，聽感還沒評；都不支援中文。

---

## 2026-11-03：Pipeline Round 1（STT → LLM → TTS 整段流程，板子 GPU）
- 設定：暫存腳本（之後整理為 configs/assistant.yaml、library/runner/assistant.py、scripts/10_assistant_live.py）
- 範圍：英文 10 題助理問句（Supertonic M2 聲音合成、重取樣成 16 kHz）+ 1 題暖機；7 組設定，每組在新的程式中從閒置開始量
- 流程：granite-470m（PyTorch GPU）→ LLM 串流 → 每產生完一句交給 Supertonic 3（GPU、補齊 96 / 128，F1 聲音，獨立程式）；
  LLM greedy、最多 96 token、不開 thinking，系統提示要求一到三句口語回答；時間起點 = 音檔送進 STT（不含 VAD 講完判斷與音效卡）
- 記憶體：tegrastats 每 0.5 秒記錄整塊板子，依載入順序拆段；各段分界受量測時間點影響（同一設定重跑可差約 0.4 GB），峰值較可靠

### 觀察
| 設定 | LLM token/s | 開始出聲（中位數，範圍） | 全部合成完 | 全部播完 | 記憶體峰值 |
|---|---|---|---|---|---|
| Qwen3.5-0.8B bf16 · transformers · granite fp32 | 4.2 | 15.82 s（8.37–32.83） | 15.82 s | — | 7,171 MB |
| Qwen3.5-0.8B bf16 · transformers | 7.2 | 2.93 s（2.17–6.57） | 3.81 s | 9.11 s | 6,520 MB |
| Qwen3-0.6B bf16 · transformers | 8.9 | 2.24 s（1.57–3.92） | 2.43 s | 7.64 s | 6,352 MB |
| Qwen3-0.6B Q8_0 · llama.cpp | 48.4 | 0.87 s（0.63–1.79） | 0.87 s | 6.07 s | 5,813 MB |
| **Qwen3-1.7B Q4_K_M · llama.cpp** | **32.3** | **0.87 s（0.62–1.54）** | **0.87 s** | **4.34 s** | **6,413 MB** |
| Qwen3-1.7B · llama.cpp · 第一段遇逗號先送 | 32.2 | 0.85 s（0.63–1.54） | 0.85 s | — | 6,347 MB |
| Qwen3.5-2B Q4_K_M · llama.cpp | 24.5 | 2.04 s（1.01–2.64） | 2.48 s | 10.62 s | 7,126 MB |

- 全部播完 = 開始出聲 + 回覆音訊長度 + 播放中斷（由 rows.json 計算；所有設定都沒有中斷），主要由回覆長短決定。
- **granite fp32**：三個模型同時放時板子記憶體升到 7.17 / 7.48 GB，系統在搬記憶體，CPU 與 GPU 大多閒置，每一段都慢數倍；fp16 正常。
- **transformers 的 LLM 很慢與架構無關**：Qwen3.5（3/4 為 linear attention、加速套件未安裝）7.2 token/s，一般架構的 Qwen3-0.6B 也只有
  8.9 token/s；執行時 GPU 平均使用率 22–31%，一個 CPU 核心滿載（每個 token 的 Python / kernel 啟動負擔在板子 CPU 上很重）。
  同一個 Qwen3-0.6B 改用 llama.cpp v0.8.0（CUDA、sm_87）後 48.4 token/s、第一個 token 0.03 s，記憶體少約 0.5 GB。
- Qwen3-1.7B（llama.cpp）：開始出聲 0.87 s ≈ STT 0.09 + LLM 第一句 0.42 + TTS 第一句 0.34（各段中位數）。
- 逗號先送：第一段送給 TTS 提早（例：0.52 → 0.19 s），但 LLM 仍在產生、與 TTS 互搶 GPU，TTS 變慢（0.56 → 0.91 s），幾乎抵銷。
- Qwen3.5-2B：回覆較長，第一句常超過補齊長度（96 格），TTS 沒補齊而變慢；峰值只剩約 0.36 GB。
- 回答品質（7 題人工判讀）：Qwen3-1.7B 最穩（提醒、180 分鐘、arigatou 正確；月球「永久黑暗面」為常見誤解）；
  Qwen3.5-2B 算錯「3 小時 = 1800 分鐘」、說紅光散射較多；0.6B / 0.8B 多題錯誤。設計時器、關燈這類指令四個模型都不能真的執行。

### 結論
- 建議組合：granite-470m fp16 + Qwen3-1.7B Q4_K_M（llama.cpp）+ Supertonic 3（補齊 96 / 128）：開始出聲中位數 0.87 s，
  峰值 6.4 GB（剩約 1.1 GB）。
- 板子上 LLM 不要用 transformers 逐 token 產生；改用 llama.cpp。
- 限制：合成語音輸入、10 題、每組只跑一次；不含 VAD 講完判斷（Round 4 建議 1000 ms）、錄音與喇叭延遲；量化對品質的影響沒有另外量。

### 下一步
- 接 USB 麥克風與喇叭，用 scripts/10_assistant_live.py 實測真人說話（含講完判斷）。
- 指令類問題需要接工具呼叫（計時器、燈光）。

---

## 2026-11-03：TTS Round 1（Supertonic 3，英文、日文）
- 設定：暫存腳本（TTS 服務之後整理為 services/tts_server）
- 範圍：supertonic 1.3.1、模型 supertonic-3（約 400 MB，不支援中文）、預設 8 步、語速 1.05、ONNX Runtime 4 執行緒；
  聲音 F1、M1 × 英文、日文各 10 句助理回覆，共 40 句
- 平台：開發機 CPU；板子 CPU；板子 GPU（NVIDIA Jetson AI Lab 的 onnxruntime-gpu 1.24.0、cuDNN 9.26）。PyPI 的 aarch64 版
  onnxruntime-gpu 1.31.1 沒有 Orin（sm_87）的 GPU 程式碼，執行時出現 `cudaErrorNoKernelImageForDevice`

### 觀察
| 平台 | 每句平均（英 / 日） | RTF（英 / 日） | 板子記憶體增加量 |
|---|---|---|---|
| 開發機 CPU | 2.06 / 2.08 s | 0.61 / 0.57 | —（RSS 550 MB） |
| 板子 CPU | 3.57 / 3.86 s | 1.07 / 1.06 | 565 MB |
| 板子 GPU | 0.92 / 0.92 s | 0.28 / 0.25 | 844 MB |
| 板子 GPU + 補齊 96 / 128 | **0.33 / 0.33 s** | **0.10 / 0.09** | 948 MB |
| 板子 GPU + 補齊 128 / 192 | 0.42 / 0.42 s | 0.13 / 0.12 | 1,003 MB |

- 原版 GPU 每句都約 0.9 s、與句長無關：ORT profiler 顯示每次輸入長度改變時 Conv 要重新準備（四個模型合計多約 0.59 s），
  回到以前出現過的長度也一樣；長度不變時同一句只要約 0.28 s。cuDNN 的 DEFAULT 演算法選擇更慢（約 2.2 s），
  `cudnn_conv_use_max_workspace=0`、`cudnn_conv1d_pad_to_nc1d=1` 沒有改善；TensorRT 需要 TensorRT 10，板子沒有，未測。
- 補齊：文字補零到 96 格、latent 補零到 128 格（約 8.9 秒音訊）後長度固定，每句約 0.33 s。duration_predictor 與 text_encoder
  沒有完全忽略補零（時長差約 1%），所以波形與原版不同；官方多句批次合成也是這樣補零，尚未做聽感確認。
- GPU 第一次合成要暖機約 1.5 s（板子上第一次執行還要編譯，約 10 s，之後有快取）。

### 結論
- 板子上 TTS 用 GPU + 補齊 96 / 128：開始出聲（第一句合成時間）約 0.33 s，記憶體約 0.95 GB。
- Supertonic 不是串流式：整句算完才輸出，回覆要一句一句交給 TTS。
- 限制：只測預設 8 步、25W 模式；補齊版待聽感確認；中文需要另找模型。

---

## 2026-11-03：Nano Round 1（Jetson Orin Nano 實機複驗，GPU）
- 設定：configs/jetson_round3.yaml（git commit：5858925；device_map 載入與 accelerate 待提交）
- 範圍：Round 3 的模型 × 方法（granite baseline / vad；whisper baseline / vad / greedy；qwen3 baseline / vad / beam）×
  10 種條件（加回 Round 1 的 snr20、snr10、snr05、reverb）× 50 句（noise_only 10 段），共 3,680 列
- 平台：Jetson Orin Nano 8GB Super、JetPack 7（L4T R39.2.1）、CUDA 13.2、PyTorch 2.15.0+cu132、25W 模式；fp32、PyTorch 直接執行
- 執行：
  - 11-02 晚 granite、whisper（2,300 列）；載入 qwen3 時舊的載入方式（先載到 CPU 再搬到 GPU）讓記憶體用滿，00:24 看門狗重開機
  - 改為 `device_map` 直接載入到目標裝置（見 decisions.md 2026-11-03）後，11-03 07:51 → 08:33 由 checkpoint 續跑 qwen3（1,380 列）；
    04_validate_results 通過；data/raw 不變
  - RAM 峰值另外量：每個模型在新程式中載入並辨識 clean 10 句，tegrastats 每 0.5 秒記錄整塊板子的已用記憶體，增加量 = 峰值 − 啟動前閒置值。
    granite 第一次量測時同時在板子上跑了結果驗證，已單獨重量（取第二次）
- Windows 參照：Round 3（i3-10110U CPU、fp32）；snr20、snr10、snr05、reverb 只有 Round 1 的 baseline

### 觀察

**WER**：所有模型 × 方法逐句輸出（正規化後）與 Windows **100% 相同**（可對照的 2,680 句），WER 完全一致。

| 模型 · 方法 | WER（Round 3 五種語音條件） |
|---|---|
| granite baseline / vad | 3.5% / 3.5% |
| whisper baseline / vad / greedy | 6.7% / 6.7% / 7.1% |
| qwen3 baseline / vad / beam | 4.7% / 4.7% / 4.4% |

**RTF**（Round 3 五種語音條件 250 句：總處理時間 ÷ 總音訊長度）

| 模型 · 方法 | Windows CPU | 板子 GPU | 加速 | 板子每句平均 |
|---|---|---|---|---|
| granite baseline | 0.348 | **0.031** | 11.2× | 156 ms |
| granite vad | 0.350 | 0.082 | 4.3× | 414 ms |
| whisper baseline | 1.996 | **0.349** | 5.7× | 1,771 ms |
| whisper vad | 2.047 | 0.394 | 5.2× | 1,997 ms |
| whisper greedy | 1.026 | 0.179 | 5.7× | 911 ms |
| qwen3 baseline | 1.006 | **0.293** | 3.4× | 1,485 ms |
| qwen3 vad | 1.006 | 0.334 | 3.0× | 1,693 ms |
| qwen3 beam | 2.911 | 0.438 | 6.6× | 2,221 ms |

- 第一句含 GPU 暖機：RTF granite 0.70、whisper 0.97、qwen3 1.56。
- 板子上 VAD 的固定成本（Silero，CPU）相對變大：granite 加 VAD 後每句多約 260 ms，RTF 由 0.031 變 0.082。

**RAM 峰值**（板子整體增加量，CPU 與 GPU 共用，含 CUDA / PyTorch 函式庫）

| 模型 | 板子增加量 | torch GPU 峰值（完整測試） | 板子載入 | Windows RSS 峰值 | Windows 載入 |
|---|---|---|---|---|---|
| granite-470m | 2,213 MB | 1,836 MB | 27.9 s | 2,291 MB | 33.1 s |
| whisper-small | 2,320 MB | 1,521 MB | 24.3 s | 3,849 MB | 50.0 s |
| qwen3-asr-0.6b | 3,665 MB | 3,112 MB（beam 3,527） | 34.7 s | 3,773 MB | 6.9 s |

- 板子閒置約 1,424 MB（tegrastats 的已用值），總共 7,485 MB；qwen3 載入後板子峰值 5,089 MB。
- 量測只用 baseline 解碼；qwen3 beam 的 GPU 峰值比 baseline 多約 400 MB。
- 量法不同：板子是系統整體增加量，Windows 是行程 RSS，只能粗略對照。

### 結論
- **換到板子 GPU 不影響辨識結果**：fp32 下逐句輸出完全相同，Windows 的 WER 結論可以直接沿用。
- **速度**：三個模型在板子上都遠快於即時（RTF < 0.5）；granite 每句約 0.16 秒，最快。
- **記憶體**：granite、whisper 約 2.2–2.3 GB，qwen3 約 3.7 GB；7.4 GB 的板子可放一個模型加喚醒詞與 VAD，
  但舊的載入方式（先 CPU 後 GPU）會讓峰值翻倍，qwen3 因此撐爆記憶體。
- 限制：fp32、PyTorch 直接執行，不是量化（q4）或 TensorRT 的部署數字；25W 模式；板子為共用。

### 下一步
- 部署版本：q4 / TensorRT 後重量 WER、RTF、RAM。

---

## 2026-11-03：Nano Round 2（Round 4 的 B、D、E 在板子 CPU 上重跑）
- 設定：板子上的 configs/round4_jetson_cpu.yaml（與 round4.yaml 相同，只改 output_dir 為 results/jetson_cpu/round4；未進版控）
- 範圍：B 講完判斷模擬（1,500 情境 × 10 組參數）、D 整段流程（200 情境 × oracle / t0.5_s700 / t0.5_s1000，granite-470m CPU）、
  E 待機成本；A 誤喚醒與 C 正確喚醒未在板子上跑
- 執行：板子 CPU（6 核 ARM、25W 模式）、fp32，2026-11-02 22:41 → 23:30（仍用 CPU 版 PyTorch）

### 觀察
- **B、D 的結果與開發機完全相同**：截斷率、失控率、等待時間、WER、ΔWER 與判定一致，規則同樣選出 t0.5_s700，
  D 中同樣顯著變差（+1.77 百分點），最終建議仍為 t0.5_s1000。
- 時間（t0.5_s1000）：granite 辨識中位數 1.68 → 2.68 秒，講完到出字中位數 2.72 → 3.68 秒（p90 3.92 → 4.56）。

| 元件 | CPU（開發機 → 板子） | 每塊處理中位數 / p99（ms） | 記憶體（MB） |
|---|---|---|---|
| 喚醒詞 kws-zh / kws-zh-en | 3.3% → 8.7% | 0.24 / 14 → 0.60 / 27 | 51 → 51 |
| Silero VAD | 1.6% → 16.3% | 0.44 / 1.2 → 1.61 / 3.0 | 167 → 196 |
| granite-470m（CPU 常駐） | — | — | 2,140 → 3,306 |

### 結論
- 講完判斷的參數與選擇不受平台影響，板子上沿用 t0.5_s1000。
- 板子 CPU 比開發機慢：VAD 的 CPU 佔用約為開發機的 10 倍（每塊 1.6 ms，仍遠小於 32 ms，跟得上）。
- 辨識放在板子 CPU 上，講完到出字約 3.7 秒；改用 GPU（Nano Round 1：granite 每句約 0.16 秒）可大幅縮短。

---

## 2026-10-31：Round 4 自動測試（喚醒詞誤喚醒、講完判斷模擬、整段流程、待機成本）
- 設定：configs/round4.yaml（git commit：A、B 為 df03544；D、E 待提交）；規格與選擇規則見 plan.md 15.3–15.4（執行前訂定；D 的規則為 v0.9，在 B 之後、D 之前訂定）
- 範圍：
  - A 誤喚醒：kws-zh、kws-zh-en × 門檻 0.15 / 0.25 / 0.35；Common Voice 英文 test 2,324 句接成一條串流（3.19 小時）、DEMAND 4 種噪音全段（20 分鐘）
  - B 講完判斷：調參集 50 句 × 開口延遲 2 × 停頓 5 × 背景 3 = 1,500 情境 × 10 組參數（threshold 0.3 / 0.5 × 靜音 300–1500 ms）
  - C 正確喚醒：尚未錄音，略過
  - D 整段流程：測試集 50 句 × 開口延遲 0.5 s × 單句或停頓 0.6 s × SNR 30 / 10 dB = 200 情境 × oracle / t0.5_s700 / t0.5_s1000，granite-470m
  - E 待機成本：kws-zh、kws-zh-en、Silero VAD、granite-470m，各在獨立子行程中量；語音、噪音各 300 秒
- 執行：開發機 CPU，2026-10-31 19:53 → 20:42（A 約 44 分鐘、B 約 5 分鐘）；22:14 E（約 3 分鐘）；22:18 → 22:38 D（單獨執行）；data/raw 不變
- 摘要：`uv run python scripts/08_wake_eval.py --config configs/round4.yaml --summary-only`

### 觀察

**A：誤喚醒**

- 兩個模型、三個門檻，在 3.19 小時英文與 20 分鐘噪音中**都是 0 次誤喚醒**（kws_negative_hits.csv 無資料）。
- 0 次的 95% 上界約 0.94 次 / 小時（3 / 3.19）；門檻之間沒有差異，無法用負樣本分出優劣，需等正樣本。

**B：講完判斷**（截斷率 = 語音還沒結束就判定講完；等待 = 講完到判定的時間）

| 參數 | 截斷率：停頓 0.3 / 0.6 / 0.9 / 1.2 s | 單句截斷 | 適用範圍截斷率 | 等待中位數（p90） | SNR 0 失控率 |
|---|---|---|---|---|---|
| t0.5_s300 | 51% / 98% / 98% / 98% | 7.7% | 53.7% | 0.31 s（0.44） | 0.6% |
| t0.5_s500 | 5.0% / 88% / 94% / 96% | 0.3% | 32.8% | 0.51 s（0.67） | 0.8% |
| **t0.5_s700（依規則選定）** | 2.3% / 7.7% / 91% / 94% | 0% | **2.2%** | **0.71 s（0.86）** | 1.6% |
| t0.5_s1000 | 1.7% / 1.7% / 4.7% / 88% | 0% | 0.5% | 1.03 s（1.18） | 2.2% |
| t0.5_s1500 | 1.3% / 1.0% / 1.3% / 1.7% | 0% | 0.5% | 1.50 s（1.68） | 3.8% |
| t0.3_s700 | 1.3% / 2.0% / 83% / 87% | 0% | 0.7% | 0.74 s（1.20） | 2.8% |

- 適用範圍 = 單句或停頓 ≤ 0.6 s、背景 SNR 30 / 10 dB；所有參數在此範圍的失控率都是 0%。
- 靜音長度必須大於句中停頓，否則幾乎一定截斷：停頓 0.9 s 要 1,000 ms 以上、1.2 s 要 1,500 ms。
- 等待時間約等於靜音長度（Silero 對句尾判斷很準），p90 多 0.15～0.5 s。
- threshold 0.3 截斷較少，但 SNR 0 dB 時較容易把噪音當語音（失控率較高）。
- 漏收率：所有參數只有 SNR 0 dB 的 1 個情境（0.2%）。

**D：整段流程**（測試集；oracle = 在實際語音結束處切，不受截斷影響的對照）

| 講完判斷 | WER | 單句 | 停頓 0.6 s | ΔWER（95% CI） | 變好 / 變差 / 不變 | 截斷率 | 辨識（中位數） | 講完到出字（中位數 / p90） |
|---|---|---|---|---|---|---|---|---|
| oracle | 2.66% | 2.94% | 2.51% | — | — | 0% | 1.48 s | — |
| t0.5_s700 | 4.43% | 2.37% | 5.46% | +1.77（+0.19, +3.70）**顯著變差** | 8 / 8 / 184 | 3.0% | 1.58 s | 2.30 / 3.39 s |
| **t0.5_s1000** | 2.62% | 2.37% | 2.75% | −0.03（−0.41, +0.43）無顯著差異 | 9 / 4 / 187 | 1.0% | 1.68 s | 2.72 / 3.92 s |

- 700 ms 在測試集的截斷率 3%（6 個情境，全部是停頓 0.6 s，其中 5 個在 SNR 10 dB），但一次截斷就丟掉後半句：SNR 10 dB 時刪除錯誤 11 → 57 個。
  B 在調參集算的截斷率（2.2%）看起來低於 5%，但截斷率不能反映「每次截斷損失多少詞」。
- 1,000 ms 與 oracle 無差異；單句時兩者都比 oracle 略好（2.37% vs 2.94%），oracle 只在語音結束後留 50 ms，句尾可能被切到一點。
- 講完到出字：等待約 1.0 秒 + granite 辨識約 1.7 秒（開發機 CPU、音訊平均 7 秒）。

**E：待機成本**（開發機 CPU；CPU = 即時處理時佔一顆邏輯核心的 %）

| 元件 | CPU | 每塊處理時間（中位數 / p99） | 載入後增加的記憶體 |
|---|---|---|---|
| 喚醒詞 kws-zh / kws-zh-en（1 緒，每 100 ms） | 3.3% / 3.2% | 0.24 / 約 14 ms | 51 MB |
| Silero VAD（每 32 ms） | 1.6% | 0.44 / 1.2 ms | 167 MB（大多為 torch） |
| granite-470m（fp32，常駐） | — | — | 2,140 MB |

- 語音與噪音的成本相同。喚醒詞 p99 約 14 ms：模型每累積數個 chunk 才解碼一次，仍遠小於 100 ms，不會跟不上。

### 結論
- **喚醒詞**：英文對話與家庭噪音下 3 小時 0 次誤喚醒，兩個模型都可用；門檻與模型要等正樣本（自錄音）決定。
  待機只佔一顆核心約 3%、記憶體約 50 MB。
- **講完判斷：最終建議 threshold 0.5、靜音 1,000 ms**。B 依規則選出 700 ms，但 D 在測試集上顯著變差，依 plan.md 15.4 改用 1,000 ms：
  WER 與 oracle 無差異，講完約 1 秒判定，講完到出字中位數 2.7 秒（開發機）。
- 方法上的教訓：只看截斷「次數」會低估影響，選講完判斷參數時要看截斷後的 WER。
- 限制：負樣本只有英文，沒有測旁人說中文；模擬的停頓是兩句拼接，不等於真實說話的猶豫；
  整段流程不含喚醒詞尾巴與回覆聲；時間為開發機 CPU fp32，不代表 Jetson。

### 下一步
- 錄音（07_record_wake.py）後跑正確喚醒（`08_wake_eval.py --parts positive`），選模型與門檻。
- 手動實測（09_live_demo.py）：手動實測的預設靜音已改為 1,000 ms（round4.yaml endpoint.default），確認真實對話中是否會截斷或等太久。
- 選用：以 Common Voice zh-TW 補中文負樣本。

---

## 2026-10-31：Round 3（VAD 開關用法驗證、解碼方法 beam / greedy）
- 設定：configs/round3.yaml（git commit：baseline、vad 為 e117214；beam、greedy 與 granite × vad 重跑為 db71bf0，meta.json 記錄後者）
- 範圍：granite-470m、whisper-small、qwen3-asr-0.6b × clean、snr00、snrm05、snrm10、lowvol、noise_only × 50 句（noise_only 10 段），共 2,080 列
  - A：baseline、vad（`mode: gate`、Silero threshold 0.3；threshold 依 VAD 參數掃描的調參資料選定）
  - B：解碼方法 beam（qwen：greedy → beam 5）、greedy（whisper：beam 5 → greedy）；
    排除 granite × beam / greedy（CTC 無解碼參數）、whisper × beam、qwen × greedy（與 baseline 相同）
- 執行：開發機 CPU fp32，2026-10-31 13:24 → 15:52（A）、16:56 → 18:22（B）；04_validate_results 通過；data/raw 不變
  - granite × vad 第一次執行時同機跑 pytest，延遲受干擾；已移除該 260 列並於 16:07 單獨重跑（WER 與轉寫和第一次完全相同）
- 摘要：`uv run python scripts/05_quick_report.py results/round3`

### 觀察

**A：VAD 開關用法（gate）**

| 模型 | WER baseline → vad（全部語音條件） | snrm10 | 幻覺率（noise_only） | VAD 每句成本（中位數） |
|---|---|---|---|---|
| granite-470m | 3.53% → 3.53% | 6.26% → 6.26% | 10% → 10% | 77 ms |
| qwen3-asr-0.6b | 4.67% → 4.67% | 7.78% → 7.78% | 100% → 10% | 129 ms |
| whisper-small | 6.68% → 6.68% | 12.33% → 12.33% | 100% → 10% | 134 ms |

- 三個模型所有語音條件的轉寫都與 baseline 相同：threshold 0.3 沒有擋掉任何一句語音（推算時 snrm10 有 1 句會被擋，實際沒有）。
- 剩下的 10% 幻覺仍是 noise10（類似音樂的片段）。
- 與 Round 2 的切除用法（trim、threshold 0.5）相比：Round 2 整體 WER 變差 0.45～0.80 個百分點，gate 無損失。
- 延遲：granite × vad 重跑後，語音句的模型延遲中位數仍比 baseline 高 17%（1,439 → 1,691 ms），
  但送入模型的音檔與 baseline 相同；whisper +2.7%、qwen −1.4%。原因未確認（可能與 CPU 頻率或同程序載入 Silero 有關）。
  05_quick_report 的平均延遲包含 noise_only（gate 擋下時不呼叫模型），兩者相近，不能用來判斷。

**B：解碼方法**

| 模型 × 方法 | WER（全部語音條件） | ΔWER（95% CI） | 變好 / 變差 / 不變 | snrm10 | 延遲（平均） | RTF | 峰值 RAM |
|---|---|---|---|---|---|---|---|
| qwen baseline（greedy） | 4.67% | — | — | 7.78% | 5.1 s | 1.01 | 3,773 MB |
| qwen + beam（beam 5） | 4.36% | −0.30（−0.61, −0.08）**顯著改善** | 6 / 0 / 244 | 6.83% | 14.8 s | 2.91 | 4,218 MB |
| whisper baseline（beam 5） | 6.68% | — | — | 12.33% | 10.1 s | 2.00 | 3,849 MB |
| whisper + greedy | 7.10% | +0.42（−0.19, +1.05）無顯著差異 | 9 / 13 / 228 | 13.09% | 5.2 s | 1.03 | 1,518 MB |

- qwen beam 5：沒有任何一句變差，改善集中在 snrm10（−0.95）；但延遲為 2.9 倍，RAM 多約 450 MB。
- whisper greedy：延遲減半；WER 增加主要在 snr00（+1.52），不顯著。noise_only 幻覺率 100% → 90%。
- 解碼方法不影響幻覺：qwen + beam 仍為 100%，需搭配 VAD（本輪未測 vad+beam 組合）。

### 結論
- **VAD 應以開關用法部署**：WER 零損失，qwen / whisper 的純噪音幻覺 100% → 10%。這是目前唯一可直接採用的改善。
- **解碼參數是「準確率換速度」**：qwen beam 5 是三輪中唯一顯著降低 WER 的方法（相對 −6.5%），但延遲 2.9 倍，
  不適合即時對話；whisper 改 greedy 延遲減半、WER 損失不顯著，部署時 greedy 較合適。
- 三輪合計：訊號層前處理（降噪、loudnorm）與解碼參數對這三個模型的 WER 改善都很小；
  granite（3.53%、RTF 0.35）仍比 qwen + beam 更準且快約 8 倍，選模型的影響大於調方法。
- 單一條件 50 句（約 500 詞，1 個錯誤 ≈ 0.2 個百分點），偵測不到 10% 左右的相對改善。

### 下一步（待討論，尚未決定）
- Round 4：喚醒詞（sherpa-onnx KWS，「小咪小咪」）與講完判斷（VAD endpointing）；
  先做不需錄音的部分（以 Common Voice 當負樣本測誤喚醒、模擬斷句截斷率與延遲），正樣本需實機錄音。
- 文字層方法（qwen 上下文提示 / 熱詞）需領域詞彙清單，現有 Common Voice 句子沒有專有名詞可測。

## 2026-10-31：VAD 參數掃描（原 Round 3，已停止）
- 設定：configs/vad_tune.yaml、configs/conditions_tune.yaml
- 範圍：granite-470m × 12 組 VAD 設定（不用 VAD、Silero 7 組、WebRTC 4 組）
  × clean、snrm05、snrm10 × 50 句調參資料（女 25、男 25；seed 7，排除測試集說話者），
  加上 noise_only 20 段（取自噪音疊加段，不用測試集的 holdout）
- 執行：開發機 CPU，2026-10-31 10:49 → 11:40；結果 results/vad_tune/sweep.csv
- 摘要：`uv run python scripts/06_vad_sweep.py --config configs/vad_tune.yaml --summary-only`
- 選擇規則（執行前訂定）：噪音誤放行 ≤ 預設、clean WER ≤ 預設 + 0.5，其中 snrm10 WER 最低者

### 觀察

| 設定 | clean | snrm05 | snrm10 | snrm10 刪除數 D | 噪音誤放行 |
|---|---|---|---|---|---|
| 不用 VAD | 2.92% | 4.74% | 8.94% | 27 | — |
| Silero 預設（threshold 0.5、pad 200 ms） | 3.28% | 6.02% | 17.15% | 72 | 0% |
| Silero threshold 0.3 | 2.92% | 5.84% | 16.42% | 61 | 0% |
| Silero threshold 0.2 | 2.92% | 5.84% | 14.05% | 53 | 5% |
| Silero threshold 0.1 | 3.28% | 4.93% | 12.59% | 41 | 30% |
| Silero threshold 0.3 + pad 400 ms（規則選出） | 2.74% | 5.47% | 15.69% | 55 | 0% |
| WebRTC a0～a2 | 2.92～3.10% | 5.11～6.39% | 9.12～9.67% | 27 | 75～85% |
| WebRTC a3 | 3.47% | 11.31% | 25.36% | 109 | 40% |

- Silero 只要噪音誤放行維持 0%，snrm10 WER 仍為不用 VAD 的近兩倍；降低 threshold 可減少切除，但噪音開始放行。
- WebRTC a0～a2 幾乎不切語音，但 75% 以上的純噪音段被放行，無法抑制幻覺。
- 本調參資料中男女聲在 snrm10 都變差（預設：女 17.36%、男 16.96%），Round 2「只有女聲變差」未重現，
  切除與否取決於句子與噪音片段，而非性別。
- 推算「VAD 只當開關」（偵測到任何語音就送整段原音，否則輸出空白；由不用 VAD 的逐句結果推算，未實際執行）：
  Silero threshold 0.3 的 snrm10 WER 為 9.85%（1 句整句未偵測到），clean 與 snrm05 與不用 VAD 相同，噪音誤放行 0%。

### 結論
- VAD 參數調整無法解決低 SNR 的語音切除；問題在「切片後串接」的用法，而非參數。
- 機器人部署（喚醒詞後由 VAD 判斷講完）應採「開關／斷句」用法，不切除句中片段；斷句的截斷率與延遲留待部署階段驗證。
- VAD 對語音條件的 WER 本來就沒有改善（Round 1、2），其效益只在抑制純噪音的幻覺；
  研究目標是降低 WER，故停止 VAD 方向，不做原定的測試集驗證。

### 下一步
- Round 3 改測解碼層與文字層的方法（束搜尋、熱詞／上下文提示等），針對錯誤分析中的替換錯誤與專有名詞。

## 2026-10-31：Round 2（負 SNR、loudnorm、較溫和的降噪）
- 設定：configs/round2.yaml（git commit：d453833）
- 範圍：granite-470m、whisper-small、qwen3-asr-0.6b × baseline、vad、denoise、loudnorm、vad+loudnorm、vad+denoise
  × clean、snr00、snrm05、snrm10、lowvol、noise_only × 50 句（noise_only 10 段），共 4,680 列
- 與 Round 1 的差異：新增 snrm05 / snrm10；denoise 為 noisereduce `prop_decrease: 0.5`（Round 1 為預設 1.0）；
  不含 snr20 / snr10 / snr05 / reverb。兩輪「全條件 WER」的條件集合不同，不可直接比較
- 執行：開發機 CPU fp32，2026-10-31 02:58 → 09:58（7 小時）；04_validate_results 通過；data/raw 不變
- 摘要：`uv run python scripts/05_quick_report.py results/round2`

### 觀察

**負 SNR 下的 baseline WER**：snrm10 為 granite 6.26%、qwen 7.78%、whisper 12.33%，三個模型仍相當耐噪。

**VAD 在 snrm10 顯著變差，且只發生在女聲**

| 模型 | snrm10 baseline | + vad | 刪除數 D（baseline → vad） | 女聲 WER | 男聲 WER |
|---|---|---|---|---|---|
| granite-470m | 6.26% | 10.82%（+4.55，顯著變差） | 15 → 38 | 8.1% → 17.4% | 4.5% → 4.5% |
| qwen3-asr-0.6b | 7.78% | 11.39%（+3.61，邊界顯著，見下） | 9 → 37 | 7.4% → 15.5% | 8.2% → 7.4% |
| whisper-small | 12.33% | 15.37%（+3.04，無顯著差異） | 4 → 31 | 17.4% → 24.0% | 7.4% → 7.1% |

- 增加的錯誤幾乎都是刪除：Silero VAD 在 −10 dB 下把部分女聲語段判為非語音而切掉。
- 整句被切光的只有每個模型 1 句，大多是部分切除。
- 含 vad 的組合（vad+loudnorm、vad+denoise）在 snrm10 也都變差，幅度相近。
- **顯著性的穩定度**：單一條件只有 50 句，CI 很寬。以結果網頁的另一組亂數重抽（同為 1000 次）重算時，
  qwen 的 vad 與 vad+loudnorm（snrm10，CI 下限 +0.19）及 whisper 的 vad+denoise（snrm10，下限 +0.17）
  的下限落到 0 附近，判定變為無顯著差異；Round 1 的 qwen vad+denoise（snr10，下限 +0.18）亦同。
  這幾格應視為「邊界顯著」。granite 的 vad（snrm10，下限 +0.95）與整體（all）的判定不受影響。

**較溫和的降噪：不再傷害，但仍無改善**
- 整體 ΔWER：granite 0.00、whisper +0.15、qwen +0.04，皆無顯著差異（Round 1 預設參數：whisper +3.88、qwen +1.08，顯著變差）。
- 各條件（含 snrm10）均無顯著改善。

**loudnorm：無效果**
- lowvol 的 ΔWER：granite −0.19、whisper −0.57、qwen −0.19，皆無顯著差異；整體 ΔWER 為 −0.04～+0.15。
- 幻覺率：loudnorm 本身不降低幻覺（whisper 100%、qwen 80%）；vad+loudnorm 與 vad 相同（10%）。

**幻覺率**：與 Round 1 一致，vad 使 whisper 與 qwen 由 100% 降到 10%（剩下的仍是 noise10）；
granite + denoise 為 0%（noise10 經降噪後不再輸出）。

**其他**
- whisper-small 的 vad+loudnorm 在 clean 顯著變差（+1.33），vad+denoise 在 clean 也顯著變差（+0.95）；
  原因未查（推測與 VAD 切除後語句邊界變化有關）。
- Exp0：假錯誤占未正規化 WER 的 30～41%，與 Round 1 一致。

### 結論
- **Exp1（VAD）**：VAD 的價值只在抑制無語音時的幻覺；在極低 SNR（−10 dB）會切掉女聲語段，WER 顯著變差。
- **Exp1（loudnorm）不成立**：三個模型對 −20 dB 音量已足夠穩健，loudnorm 無可量測效果。
- **Exp2（降噪）不成立**：noisereduce 預設參數有害，減半壓制後無害但仍無改善，−10 dB 下亦然。
- 綜合兩輪：這三個模型對穩態環境噪音與低音量相當穩健，訊號層前處理的改善空間很小；
  VAD 應保留以抑制幻覺，但需處理極低 SNR 時的漏切。

### 下一步（待討論，尚未決定）
- 見對話中提出的 Round 3 方向：VAD 參數（threshold、pad）在低 SNR 的調整、文字層 / 解碼層方法（plan.md 第 2 章延後項目）、
  更真實的噪音（人聲干擾、實機錄音）、Jetson q4 部署驗證。

## 2026-10-31：Round 1（Exp0 正規化、Exp1 VAD、Exp2 降噪）
- 設定：configs/round1.yaml（git commit：0917be6；結果 commit a3c0a72）
- 範圍：granite-470m、whisper-small、qwen3-asr-0.6b × baseline、vad、denoise、vad+denoise × 8 條件 × 50 句（女 25、男 25；noise_only 為 10 段），共 4,320 列
- 執行：開發機 i3-10110U、CPU fp32、4 執行緒，2026-10-30 18:19 → 10-31 00:45（6 小時 26 分）；
  04_validate_results 通過；data/raw 檔案數與大小不變
- 摘要：`uv run python scripts/05_quick_report.py results/round1`

### 觀察

**整體 ΔWER（全部語音條件，相對同模型 baseline，百分點；配對 bootstrap 95% CI）**

| 模型（baseline WER） | vad | denoise | vad+denoise |
|---|---|---|---|
| granite-470m（2.85%） | 0.00 [−0.30, +0.34] 無顯著差異 | +0.30 [−0.05, +0.66] 無顯著差異 | +1.14 [+0.57, +1.78] 顯著變差 |
| whisper-small（5.42%） | +0.11 [−0.44, +0.71] 無顯著差異 | +3.88 [+2.79, +5.19] 顯著變差 | +3.55 [+2.47, +4.66] 顯著變差 |
| qwen3-asr-0.6b（3.74%） | −0.05 [−0.43, +0.40] 無顯著差異 | +1.08 [+0.41, +1.83] 顯著變差 | +1.82 [+1.09, +2.69] 顯著變差 |

- 一句話結論（05_quick_report）：qwen3-asr-0.6b + vad，Δ −0.05，無顯著差異。此句依定義取 Δ 最負者；
  實際意義為**本輪沒有任何方法顯著降低 WER**。
- 各條件中顯著的結果全部是「變差」，沒有任何條件顯著改善：
  - reverb：denoise 在三個模型都顯著變差（granite +1.52、qwen +4.93、whisper +13.47）。
  - whisper-small：denoise 在 clean 也顯著變差（+2.28），表示 noisereduce 本身會引入失真，而不只是在噪音條件下無效。
- 劣化條件難度偏低：snr00 下 baseline WER 只有 3.04%（granite）～5.31%（whisper），
  DEMAND 噪音對這三個模型影響小，前處理可改善的空間有限。
- 語音條件的錯誤以替換（S）為主，插入（I）極少（baseline：granite 1、qwen 12、whisper 13 個，N = 3,689），
  因此 VAD 對語音條件 WER 沒有可量測的影響。

**Exp1 幻覺率（noise_only 輸出非空的比例）**

| 模型 | baseline | vad | denoise | vad+denoise |
|---|---|---|---|---|
| granite-470m | 10% | 10% | 10% | 10% |
| qwen3-asr-0.6b | 100% | 10% | 90% | 10% |
| whisper-small | 100% | 10% | 100% | 10% |

- whisper baseline 在純噪音上輸出 "Thank you so much for watching!" 等句子；whisper + denoise 在 noise05 輸出上百次重複的 "I'm sorry"。
- qwen baseline 幾乎都輸出 "The."。
- VAD 後剩下的 10% 全部是 noise10（M2 試聽時判斷為音樂類噪音）：Silero VAD 將其判為語音而未濾除。
  granite 只在 noise10 有輸出，與方法無關。

**Exp0 正規化影響（baseline）**

| 模型 | 未正規化 WER | 正規化 WER | 假錯誤 |
|---|---|---|---|
| granite-470m | 4.29% | 2.85% | 1.44 百分點（33.6%） |
| whisper-small | 9.34% | 5.42% | 3.92 百分點（42.0%） |
| qwen3-asr-0.6b | 6.98% | 3.74% | 3.24 百分點（46.4%） |

**其他**
- 男聲 WER 在三個模型都高於女聲（baseline：granite 3.77 vs 1.88、qwen 5.52 vs 1.88、whisper 6.59 vs 4.21）。
- 延遲（開發機 CPU，僅供方法間相對比較）：baseline RTF 為 granite 0.24、qwen 0.96、whisper 1.91（RTF > 1 表示比即時慢）。
- whisper-small baseline 峰值 RAM 3,915 MB，明顯高於其他方法（約 2,116 MB）；baseline 為該模型第一個方法，
  推測含模型載入時的暫時高峰，比較方法間 RAM 時需留意。

### 結論
- **Exp0 成立**：未正規化 WER 中有 34～46% 是假錯誤，正規化為必要步驟；所有 WER 皆以正規化後計算。
- **Exp1 部分成立**：VAD 不改變語音條件的 WER（插入錯誤本來就極少），但將 whisper 與 qwen 的幻覺率由 100% 降到 10%，
  是本輪唯一有明確效益的方法。loudnorm 未納入本輪，「loudnorm 改善 lowvol」尚未驗證。
- **Exp2 不成立**：noisereduce 在任何 SNR 都沒有顯著改善，在 reverb 與（whisper 的）clean 顯著變差；
  vad+denoise 的效果由 denoise 主導。此結論僅代表 noisereduce（頻譜閘控、預設參數），不代表 DeepFilterNet。

### 下一步（待討論，尚未決定）
- loudnorm 尚未測試：Round 2 加入 `loudnorm`（及 `vad+loudnorm`）以驗證 lowvol 假設。
- 降噪：調整 noisereduce 參數（例如降低 prop_decrease），或在相容版本可用時改測 DeepFilterNet。
- 劣化難度：目前條件下 baseline WER 偏低；若要觀察方法在困難情境的效果，需更低 SNR 或其他噪音類型
  （新增條件名稱需先修改 plan.md 6.5 命名規則）。
- noise10：檢查 VAD threshold 對音樂類噪音的敏感度。
- 部署：在 Jetson Orin Nano 以 q4 重新驗證 ΔWER 與資源。
