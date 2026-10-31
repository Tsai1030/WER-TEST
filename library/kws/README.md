# library/kws/

喚醒詞偵測（sherpa-onnx KWS，Apache 2.0）。喚醒詞以建模單元設定（中文：聲母 + 帶聲調韻母），不需訓練或聲音向量。

| 檔案 | 做什麼 |
|---|---|
| spotter.py | KeywordSpotter：load() / spot(wav, sr) 整段串流偵測 / stream() 即時串流 / unload()；Detection（keyword、t_sec）；keywords_text()、missing_tokens() |

## 範例
    s = KeywordSpotter(spotter_config(cfg, "kws-zh-en", 0.25)); s.load()
    hits = s.spot(wav, 16000)          # [Detection("小咪小咪", 1.6)]
    st = s.stream(); st.accept(chunk)  # 麥克風串流，每次送入 100 ms

## 喚醒詞寫法
`{ tokens: "x iǎo m ī x iǎo m ī", label: 小咪小咪 }`：tokens 必須都在模型的 tokens.txt 中（load() 會檢查）。
中文以「聲母 韻母」拆開並標聲調（ī 為一聲、ǐ 為三聲），例：小米小米 = `x iǎo m ǐ x iǎo m ǐ`。

## 模型下載（不進版控，放 .kws_models/）
    https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2   （中文）
    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2            （中英混合）
解壓到 `.kws_models/<模型名稱>/`；使用的檔名見 configs/round4.yaml 的 kws.models。

## 注意
- 安裝：`uv sync --extra audio --extra models --extra wake`。
- keywords_threshold 越高越不容易誤喚醒，也越容易漏喚醒；keywords_score 越高越容易觸發。
