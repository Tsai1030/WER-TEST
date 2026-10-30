# library/textnorm/

文字正規化。ref 與 hyp 必須使用同一個 normalizer。

| 檔案 | 做什麼 |
|---|---|
| english.py | 小寫、去標點、縮寫展開、數字統一、英美拼法統一、移除贅詞 |
| registry.py | get_normalizer(name)；預留中文（簡轉繁、全半形、數字）版本 |

## 範例
    normalize("I don't know, uh, twenty-five.")  # → "i do not know 25"
