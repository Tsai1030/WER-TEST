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
