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

## 2026-11-02：Jetson 改用 GPU 版 PyTorch；新增 Jetson 複驗設定
- 決定：`pyproject.toml` 的 torch 來源依平台區分：Linux aarch64 用 PyTorch 官方 cu132 index
  （`2.15.0+cu132`），其他平台維持 CPU index（開發機仍為 `2.15.0+cpu`，版本號相同）；torchaudio 一律 CPU 版。
  新增 `configs/jetson_round3.yaml`：與 round3.yaml 相同的模型、方法、50 句，`device: cuda`、fp32，
  條件加回 Round 1 的 snr20 / snr10 / snr05 / reverb（共 10 種），輸出到 `results/jetson/round3/`。
- 理由：板子為 Jetson Orin Nano 8GB Super，JetPack 7（L4T R39.2.1）、CUDA 13.2；官方 cu132 有
  cp311 aarch64 版，與 CUDA 版本一致，且 torch 版本可與開發機相同。使用者同意改用 GPU（原規定 PyTorch 維持 CPU index）。
  先維持 fp32，讓 Windows 與 Jetson 的差異只來自平台；q4 部署版本另行測試（11.5）。
- 替代方案：社群編譯的 JetPack 7.2 wheel（非官方、torch 2.12）；Jetson AI Lab 的 jp6 index（JetPack 6，不相容）。
- 影響範圍：`pyproject.toml`、`uv.lock`（新增 nvidia-* CUDA 套件，只裝在 Linux aarch64）、`CLAUDE.md`、
  `configs/jetson_round3.yaml`。官方 wheel 未內建 sm_87 專用核心，以 sm_80 核心執行（相容）。

## 2026-10-31：講完判斷改用靜音 1,000 ms；新增整段流程與待機成本測試（plan.md v0.9）
- 決定：
  - 新增 D 整段流程（測試集情境 → 講完判斷 → granite 辨識，與 oracle 比較 WER 與講完到出字時間）與
    E 待機成本（各元件的即時 CPU 佔用、處理時間與記憶體）；D 的規則在執行前寫入 plan.md 15.4。
  - 講完判斷最終採用 threshold 0.5、靜音 1,000 ms（round4.yaml endpoint.default 改為 1000，手動實測預設使用）。
- 理由：B 依規則選出 700 ms（調參集截斷率 2.2%），但 D 在測試集上 ΔWER +1.77 個百分點（95% CI +0.19～+3.70，顯著變差）：
  截斷雖只有 3%，每次都丟掉後半句。依 15.4 改用 1,000 ms，ΔWER −0.03（無顯著差異），多等約 0.3 秒。
- 替代方案：1,500 ms（停頓 0.9 s 也不截斷，但每句多等 0.5 秒，D 未測）；threshold 0.3 + 700 ms（B 的截斷率較低，但 SNR 0 dB 失控率較高，D 未測）。
- 影響範圍：configs/round4.yaml（pipeline、standby 區段，endpoint.default）、library/runner/wake.py、library/monitor/cpu.py、
  scripts/08_wake_eval.py、results/round4/pipeline.csv、standby.csv、結果網頁 Round 4 分頁。

## 2026-10-31：Round 4：喚醒詞（sherpa-onnx KWS）與講完判斷（plan.md v0.8 第 15 章）
- 決定：
  - 情境：中文喚醒詞「小咪小咪」→ 回覆「我在我在」→ 使用者說英文（可能不等回覆說完就開口）→ 講完判斷 → granite 辨識；
    喚醒後可連續多句，沒說話逾時才回到等待喚醒。
  - 喚醒詞引擎用 sherpa-onnx KWS（Apache 2.0）：以拼音設定喚醒詞，不需訓練或聲音向量；
    比較兩個模型：中文 `wenetspeech-3.3M`（2024-01）與中英混合 `zh-en-3M`（2025-12），門檻掃描 0.15 / 0.25 / 0.35。
  - 講完判斷用 Silero 串流 VAD 的狀態機（新模組 library/endpoint/），掃描 threshold 0.3 / 0.5 × 靜音 300–1500 ms。
  - 偵測到喚醒詞就開始收音（回覆播放期間繼續收音，可搶話）；收音從喚醒判定前 lookback_ms 開始。
  - 自動測試（不需錄音）：
    A 誤喚醒：Common Voice 英文 3 小時首尾相接成一條串流 + DEMAND 噪音 20 分鐘，算每小時誤喚醒次數；
    B 講完判斷：調參集 50 句組成 1,500 個「喚醒後說話」情境（開口延遲 0 / 0.5 s、單句或句中停頓 0.3–1.2 s、背景 SNR 30 / 10 / 0 dB）。
  - 需錄音的部分：07_record_wake.py 引導錄「小咪小咪」35 次（正常、快、慢、小聲、遠、接英文）、音近詞 15 次、回覆聲 1 次；
    09_live_demo.py 即時實測，可只測喚醒（--kws-only）。
  - 講完判斷的推薦規則（執行前訂定）：單句或停頓 ≤ 0.6 s、背景 SNR 30 / 10 dB 的情境中，截斷率與失控率都 ≤ 5% 的參數裡，取等待時間中位數最小者。
- 理由：
  - 部署情境需要喚醒詞，且喚醒後的收音長度由講完判斷決定；截斷會直接造成辨識刪除錯誤，等待時間決定回應速度。
  - Round 1–3 顯示 VAD 不宜切除句中片段，Round 4 的 VAD 只用來判斷開始與結束。
  - 誤喚醒的主要來源是喚醒後的英文對話與環境聲，Common Voice 英文與 DEMAND 可直接當負樣本。
- 替代方案：openWakeWord（以英文為主，中文喚醒詞需自行訓練）、Porcupine（商業授權）。
- 限制：
  - 沒有中文負樣本（台灣中文對話是實際環境的主要誤喚醒來源），只能以自錄的音近詞與中文對話少量測試。
  - 正樣本只有使用者一人、筆電或耳機麥克風，正確喚醒率不能代表其他人與機器人麥克風。
  - 外放喇叭時回覆聲會被收音，實機需要回音消除（AEC）；手動實測建議戴耳機，或加 --no-listen-during-reply。
- 影響範圍：新增 library/kws/、library/endpoint/、library/runner/wake.py、library/runner/live.py、
  scripts/07–09、configs/round4.yaml；pyproject 新增選用群組 wake（sherpa-onnx、sounddevice）；
  模型放 .kws_models/（不進版控）；自錄音檔放 data/wake/（不進版控、不可公開）。

## 2026-10-31：新增解碼方法 beam / greedy 與設定檔 exclude（plan.md v0.7）
- 決定：
  - 方法命名新增 `beam`、`greedy`（plan.md 6.5、CLAUDE.md、dashboard/data_contract.md）。解碼方法不改變音訊，
    只覆寫模型的 `beam_size`（configs/methods/beam.yaml：5；greedy.yaml：1）。
  - 設定檔新增 `exclude`，列出不執行的「模型 × 方法」；load_config 檢查未排除的組合中模型支援該解碼參數。
  - 加入 Round 3（同一個結果資料夾，共用 baseline）：qwen3-asr × beam、whisper-small × greedy。
    granite 為 CTC，generate() 即 argmax，無解碼參數可調，不納入。
- 理由：
  - 兩輪結果顯示音訊前處理無法降低 WER；錯誤以替換為主，屬解碼與用詞問題。
  - 機器人喚醒後為自由對話，熱詞清單無法事先列出，先測不需要詞彙的解碼參數。
  - whisper greedy 量測 beam 5 的效益與延遲代價（Jetson 上 greedy 較快）。
- 替代方案：
  - granite + n-gram 語言模型（pyctcdecode + KenLM）：KenLM 在 PyPI 只有原始碼，Windows 需自行編譯 C++，
    延到 Linux（Jetson / WSL）。
  - 熱詞 / 上下文提示（qwen 的 prompt、whisper 的 prompt_ids）：適合固定詞彙，自由對話價值較低，暫緩。
- 影響範圍：`library/export/schema.py`、`library/methods/decoding.py`、`library/models/base.py`、
  `whisper.py`、`qwen3_asr.py`（num_beams）、`library/runner/config.py`、`matrix.py`、`runner.py`、
  `library/export/validate.py`、`configs/round3.yaml`。

## 2026-10-31：Round 3：VAD 開關用法（gate）的測試集驗證
- 決定：
  - VAD 新增 `mode: gate`：偵測到任何語音段就送整段原音，否則輸出空白；預設仍為 `trim`，Round 1、2 可重現。
  - Round 3：3 模型 × baseline、vad（gate、Silero threshold 0.3）× Round 2 的 6 個條件。
  - threshold 0.3 依 VAD 調參資料選定（不看測試集）：gate 推算下噪音誤放行 0%（與預設相同）且 snrm10 WER 最低
    （0.3：9.85%；0.5：11.13%；0.2 以下噪音開始放行）。0.3 與 0.3 + pad 400 ms 在 gate 下相同，取參數較少者。
- 理由：
  - 機器人以喚醒詞啟動、由 VAD 判斷講完，不會切除句中片段，gate 較接近部署用法。
  - 以 Round 2 測試集結果推算，gate 保留 VAD 抑制幻覺的效益（whisper、qwen 100% → 10%），WER 接近 baseline；
    推算未實際執行，需以本輪確認。
- 替代方案：沿用 threshold 0.5（Round 2 推算的 gate 結果即為此設定，可與本輪對照）。
- 影響範圍：`library/methods/vad.py`、`configs/methods/vad.yaml`、`configs/round3.yaml`。

## 2026-10-31：停止 VAD 調參，Round 3 改測解碼層與文字層方法
- 決定：
  - VAD 參數掃描完成後停止 VAD 方向，不做原定的測試集驗證（三模型 × 選出的 VAD 設定）。
  - Round 3 改為降低 WER 的解碼層與文字層方法；具體方法與設定另行規劃，需先修改 plan.md 6.5 的方法命名。
  - VAD 掃描的工具（`impl: webrtc`、`sampling.exclude_manifest`、noise_only `segment: mix`、06_vad_sweep.py）保留，
    供部署階段的斷句驗證使用。
- 理由：
  - 研究目標是降低 WER；兩輪結果顯示 VAD 不改善語音條件的 WER，只抑制純噪音的幻覺。
  - 掃描顯示參數調整無法消除低 SNR 的切除（見 experiment_log 2026-10-31 VAD 參數掃描）。
  - 機器人以喚醒詞啟動，ASR 只在喚醒後執行，幻覺的影響範圍較小；VAD 在部署時的角色是斷句。
  - 錯誤分析：baseline 錯誤以替換為主，約 60% 為專有名詞，屬解碼與用詞問題，訊號前處理無法改善。
- 替代方案：
  - 完成 VAD 測試集驗證（約 2.3 小時）：使用者選擇直接轉向。
  - TEN VAD：Python 版無 Linux aarch64（Jetson）套件，授權附加限制，掃描前即排除。
- 影響範圍：`configs/vad_tune.yaml`、`configs/conditions_tune.yaml`、`library/methods/vad.py`、
  `library/runner/vad_sweep.py`、`scripts/06_vad_sweep.py`、`library/data/sampler.py`、`library/degrade/registry.py`。

## 2026-10-31：Round 2 範圍：負 SNR、loudnorm、較溫和的降噪
- 決定：
  - 新增條件 `snrm05`、`snrm10`（SNR −5 / −10 dB，m = minus），沿用同一組 DEMAND 噪音與每句相同的噪音片段；
    修改 plan.md 6.5 命名規則（v0.6）、CLAUDE.md、dashboard/data_contract.md 與 sections.md。
  - Round 2 矩陣：3 模型 × 6 方法（baseline、vad、denoise、loudnorm、vad+loudnorm、vad+denoise）
    × 6 條件（clean、snr00、snrm05、snrm10、lowvol、noise_only），共 4,680 筆。
  - Round 2 的 denoise 改為 noisereduce `prop_decrease: 0.5`（只壓低一半噪音），以 round2.yaml 的
    `method_params` 設定，不修改 configs/methods/denoise.yaml，讓 Round 1 保持可重現。
- 理由：
  - Round 1 在 snr00 下 baseline WER 仍只有 3.0～5.3%，方法的改善空間不足，看不出方法在困難情境是否有效。
  - Round 1 未測 loudnorm，lowvol 假設尚未驗證。
  - noisereduce 預設（prop_decrease 1.0）在 clean 與 reverb 都顯著變差，推測完全壓除會造成失真；
    0.5 為事先選定的值（未用 Round 2 資料調參），避免在測試集上挑參數。
- 替代方案：
  - 多人說話背景（babble）或殘響 + 噪音：使用者選擇只做負 SNR。
  - 在留出資料上掃描 prop_decrease：較嚴謹但需另建資料集，本輪不做。
  - Round 2 加入 reverb：whisper 需多約 1 小時，本輪不含，reverb 的結果見 Round 1。
- 影響範圍：
  - `library/export/schema.py`（CONDITION_NAMES）、`configs/conditions.yaml`、`configs/round2.yaml`。
  - `library/runner/config.py`（method_params）、`runner.py`；`degrade_set.py` 改為合併 degrade_log.csv。
  - Round 2 的 denoise 與 Round 1 的 denoise 參數不同，報告需分開註明（meta.json 的 methods 有記錄）。
  - 兩輪的條件集合不同，「全條件 WER」不可跨輪比較。
  - M2 的 SNR 驗證改為以已知語音成分精確計算（`degrade.speech_component`）：原本的 measure_snr 需估計增益，
    負 SNR 時誤差放大（snrm10 最大 1.14 dB、平均僅 0.009 dB），屬量測誤差而非混音誤差；改後各 SNR 最大誤差 ≤ 0.001 dB。
    重新產生的 clean / snr00 / lowvol / noise_only 與 Round 1 使用的音檔 SHA256 完全相同。

## 2026-10-30：M5 執行器設計
- 續跑：已完成組合 = checkpoint.json ∪ results.csv 既有列。results.csv 逐列 flush，checkpoint 每 20 列與每個
  「模型 × 方法」結束時存檔（先寫 .tmp 再取代），中斷時也會存。即使「列已寫入、checkpoint 未存」也不會重複。
  完全跑完的模型不再載入，其資源量測與描述取自 checkpoint。
- `--no-resume`：既有輸出改名為 `.bak`（不直接刪除）後從頭執行；`.bak` 已列入 .gitignore。
- 記憶體：每個「模型 × 方法」一個 ResourceMonitor（psutil 每 50ms 輪詢 RSS），量到的是含模型權重的行程峰值。
- `detected_lang` 只記錄**自動偵測**的語言；本輪 Whisper、Qwen 皆強制英文，故留空。
- round1.yaml 新增 `conditions_file`、`max_utterances`（null）、`meta`（dataset、nano_budget_mb）。
  `max_utterances` 依女、男輪流取，小規模測試也維持性別平衡。新增 `configs/smoke.yaml`（輸出 results/smoke，不進版控）。
- meta.json 另含 `seed`、`matrix`（模型、方法、條件、句數，供驗證列數）、`resumed`。
- 驗收（2026-10-30）：smoke 設定 90 列，於第 30 列強制終止行程後以同一指令續跑，補完 60 列、無重複、驗證通過。
- 已知：即使鎖定 revision，transformers 載入時仍會連線查詢檔案清單；夜間執行若擔心斷網，可設 `HF_HUB_OFFLINE=1`。

## 2026-10-30：M4 冒煙測試範圍與時間
- 決定：`tests/test_smoke_pipeline.py` 預設只跑 granite-470m × 5 個方法（baseline、vad、denoise、loudnorm、vad+denoise）
  × clean / noise_only，2 句 + 1 段純噪音；`SMOKE_ALL=1` 時跑 round1 全部模型 × 方法 × 條件，作為 M4 驗收。
- 理由：plan 第 14 章要求冒煙測試 1 分鐘內完成，但 M4 驗收要求「2 句 × 全組合」，兩者無法同時滿足。
  開發機實測光 import torch + transformers 約 20 秒、載入 granite 15–30 秒，預設版實測 62 秒，已是此機器的下限。
- pytest 停用 faulthandler：Windows 上它會把 torch / torchaudio 原生程式碼內部已處理的例外印成
  「Windows fatal exception: access violation」，測試結果正常，屬誤報。
- 建議（選用）：將 `D:\stt test\.venv` 加入 Windows Defender 排除清單，可能縮短 import 時間。

## 2026-10-30：M2 噪音源與劣化條件細節
- 噪音源：DEMAND 16kHz 版（CC BY 4.0，Thiemann, Ito, Vincent，doi:10.5281/zenodo.1227121），
  選 DKITCHEN、DLIVING、DWASHING、OOFFICE 四個室內環境，各取 ch01（300 秒）。
  壓縮檔 md5：DKITCHEN 7ffbf52d7f4699f96927846103dc8788、DLIVING 46741384d9e434a0bd8b3ec1830b6052、
  DWASHING 7e5ee9437ce9409c5f9a779b6212a240、OOFFICE 7b61cc2d182d5a654cb9c3101ddd4041。
  - 避開 OMEETING、PCAFETER：可能含人聲，會讓 noise_only 的「輸出文字」不再等於幻覺。
  - DWASHING（洗衣機）最接近機器人本體的機械噪音；之後有機器人錄音時優先替換。
- 疊加與純噪音分離：每個噪音源最後 60 秒只給 noise_only，前 240 秒用於疊加。
- 同一句在 snr20/10/05/00 使用相同噪音環境與片段（同 seed），只改變 SNR。
- 防削波：峰值超過 0.99 時整體等比例縮小，不改變 SNR。
- 殘響：RT60 0.6s、房間 5×4×3m、距離 2.5m；麥克風高 1.0m（機器人）、聲源高 1.6m（說話者），
  擺放位置以 geometry_seed 42 固定，所有句子用同一個 RIR；輸出 RMS 調回與輸入相同，避免同時改變音量。
- 純噪音音量：rms_dbfs −32。clean 語音 RMS 中位數 −21.9 dBFS，−32 約為 snr10 時的噪音大小。
- noise10（DLIVING 保留區段，原始 −56.4 dBFS，放大 24.4 dB）第 4.1–4.7 秒被 Silero VAD 判為語音；
  人工抽聽確認為**背景音樂、非人聲**，保留不重新產生。其餘 9 段與四個噪音源整段 VAD 語音比例皆為 0%。
  分析幻覺率時需注意：音樂是 Whisper 常見的幻覺誘因（例如輸出「♪」），noise10 的結果可單獨對照。
- 影響範圍：`configs/conditions.yaml`、`library/degrade/`、`library/runner/degrade_set.py`、`scripts/02_make_degraded.py`。

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
