# library/scoring/

對齊與評分。輸入必須是正規化後的文字。

| 檔案 | 做什麼 |
|---|---|
| align.py | align()：Levenshtein 對齊，回傳 S/D/I/N 與對齊序列 |
| wer.py | 逐句 WER、corpus_wer()（ΣS+D+I ÷ ΣN） |
| stats.py | paired_bootstrap()：ΔWER 95% CI；compare_per_utterance()：變好/變差/不變 |

## 範例
    r = align("i love you", "i love u")   # S=1, D=0, I=0, N=3, wer=0.333
