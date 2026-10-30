# 網頁區塊定義

結果網頁的 5 個分頁、各區塊內容與計算方式。所有數字由網頁從 `results.csv`、`resources.csv`、
`meta.json` 計算（欄位見 `data_contract.md`）。來源：`docs/plan.md` 第 9、10 章。

## 共用定義（plan.md 第 9 章）

| 指標 | 定義 |
|---|---|
| WER | (S + D + I) ÷ N，以正規化後文字計算 |
| 整體 WER | ΣS+D+I ÷ ΣN（加權，非逐句平均）；不含 noise_only |
| S / D / I 比例 | 各自 ÷ ΣN，加總等於 WER |
| ΔWER | 方法 WER − 同模型同條件 baseline WER（百分點），負值代表改善 |
| 95% CI | 以句為單位配對 bootstrap 1000 次的 2.5% 與 97.5% 分位數 |
| 結論判定 | CI 上界 < 0 → 顯著改善；CI 下界 > 0 → 顯著變差；其餘 → 無顯著差異 |
| 變好 / 變差 / 不變 | 逐句比較錯誤數（S+D+I），方法 vs baseline |
| 幻覺率 | noise_only 片段中 `hyp_norm` 非空的比例 |
| 未正規化 WER | `ref_raw` 與 `hyp_raw` 以空白切詞直接比對 |
| 假錯誤 | 未正規化 WER − 正規化 WER |
| 峰值記憶體 | 每個「模型 × 方法」執行期間 GPU 與 RAM 峰值（resources.csv） |
| 延遲 | 單句 `latency_ms`（方法處理 + 模型辨識） |
| RTF | `latency_ms / 1000 ÷ audio_sec` |

「全條件 WER」：所有語音條件（clean、snr20、snr10、snr05、snr00、reverb、lowvol）合併計算整體 WER。

---

## 分頁 1：總覽

| 區塊 | 內容 | 計算方式 |
|---|---|---|
| 一句話結論 | 改善最多的「模型 + 方法」與其 WER 變化 | 對每個 model × method（非 baseline），取全條件 ΔWER；選最小（最負）者，顯示「<model> + <method>：WER x% → y%（Δ z 百分點，結論）」。需與 `05_quick_report.py` 一致 |
| Baseline 總覽 | 各模型 clean WER、全條件 WER、幻覺率、峰值 GPU / RAM、模型大小、載入時間、平均延遲 | 只取 method = baseline；資源欄取 resources.csv 對應列；平均延遲為 latency_ms 平均（不含 noise_only） |
| 正規化影響 | 未正規化 vs 正規化 WER、假錯誤百分點與占比 | baseline、各模型、clean 與全條件；占比 = 假錯誤 ÷ 未正規化 WER |
| 實驗設定 | meta.json 內容、各模型框架與解碼設定 | 直接列出 meta.json；models 區塊展開為表格 |

## 分頁 2：方法成本效益

| 區塊 | 內容 | 計算方式 |
|---|---|---|
| 成本效益總表 | 每個 model × method：ΔWER、95% CI、Δ GPU、Δ RAM、Δ 延遲、變好 / 變差 / 不變、結論；可切換條件（含「全條件」） | ΔWER 與 CI 依共用定義；Δ GPU / Δ RAM = 方法列 − baseline 列（resources.csv）；Δ 延遲 = 平均 latency_ms 差 |
| WER 矩陣 | 模型 × 方法 × 條件 WER 熱度表 | 每格為該組合整體 WER；noise_only 欄改顯示幻覺率 |

## 分頁 3：抗噪與錯誤類型

| 區塊 | 內容 | 計算方式 |
|---|---|---|
| 抗噪曲線 | 各模型一張小圖（共用縱軸），橫軸 clean → snr20 → snr10 → snr05 → snr00，每個方法一條線 | 每點為該 model × method × condition 整體 WER |
| S / D / I 拆解 | 選定模型與條件下，各方法的錯誤組成（堆疊長條） | S/ΣN、D/ΣN、I/ΣN |
| 幻覺率 | 模型 × 方法長條圖 | condition = noise_only 的幻覺率 |

## 分頁 4：資源

| 區塊 | 內容 | 計算方式 |
|---|---|---|
| 峰值記憶體 | GPU + RAM 堆疊長條，標示 Nano 8GB 預算線（meta.json `nano_budget_mb`），超過以紅色顯示 | resources.csv；`peak_gpu_mb` 留空時只顯示 RAM |
| 平均延遲 | 模型 × 方法長條圖，另顯示 RTF | latency_ms 平均；RTF 依共用定義（不含 noise_only） |

## 分頁 5：逐句分析

| 區塊 | 內容 | 計算方式 |
|---|---|---|
| 最差 5 句 | ref / hyp 對齊，替換、刪除、插入以顏色標示 | 依選定 model × condition × method，取錯誤數（S+D+I）最多的 5 句；對齊由網頁以 Levenshtein 從 ref_norm / hyp_norm 重算 |
| 逐句瀏覽 | 依模型 / 條件 / 方法篩選，可搜尋 ref / hyp 文字 | results.csv 原始列 |

---

## Round 1 新增項目（plan.md 10.3）

- **男女聲 WER 比較**：依 `gender` 欄分組計算整體 WER（baseline 與各方法），noise_only 不計。
- **RTF**：依 `audio_sec` 欄計算，加在平均延遲區塊與 Baseline 總覽。
- **CPU only 呈現**：`peak_gpu_mb` 留空時，記憶體圖只顯示 RAM，表格中 GPU 欄顯示「—」；
  延遲圖加註「開發機 CPU 測量，僅供方法間相對比較」。

## 注意事項（plan.md 10.5）

- 網頁只放轉寫文字與分數，**不放音檔**（Common Voice 條款）。
- `speaker_id` 使用匿名 ID，不使用 Common Voice 的 client_id。
- 網頁預設為私人，分享前需確認內容。
