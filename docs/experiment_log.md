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
| qwen3-asr-0.6b | 7.78% | 11.39%（+3.61，顯著變差） | 9 → 37 | 7.4% → 15.5% | 8.2% → 7.4% |
| whisper-small | 12.33% | 15.37%（+3.04，無顯著差異） | 4 → 31 | 17.4% → 24.0% | 7.4% → 7.1% |

- 增加的錯誤幾乎都是刪除：Silero VAD 在 −10 dB 下把部分女聲語段判為非語音而切掉。
- 整句被切光的只有每個模型 1 句，大多是部分切除。
- 含 vad 的組合（vad+loudnorm、vad+denoise）在 snrm10 也都變差，幅度相近。

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
