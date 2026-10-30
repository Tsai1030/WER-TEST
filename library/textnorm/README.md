# library/textnorm/

文字正規化。ref 與 hyp 必須使用同一個 normalizer。

| 檔案 | 做什麼 |
|---|---|
| english.py | 小寫、去標點、縮寫展開、數字統一、英美拼法統一、移除贅詞 |
| registry.py | get_normalizer(name)；預留中文（簡轉繁、全半形、數字）版本 |

## 範例
    normalize("I don't know, uh, twenty-five.")  # → "i do not know 25"

## 實作
- 先套用 Whisper `EnglishTextNormalizer`（套件 whisper-normalizer）：已移除 hmm、mm、mhm、mmm、uh、um
- 再移除 `EXTRA_FILLERS`（uhm、umm、uhh、erm、hm）；ah、er、oh 可能是一般詞彙，不移除

## 已知特性
- "oh" 會被當成數字轉為 "0"（例："oh I see" → "0 i see"）；ref 與 hyp 一致處理，不會造成假錯誤
- 連續的數字詞會合併成一個數字（"one two three" → "123"），會影響 N（詞數）；ref 與 hyp 一致處理
- 縮寫字母會拆開（"U.S.A." → "u s a"）、"Mr." → "mister"、"St." → "saint"
