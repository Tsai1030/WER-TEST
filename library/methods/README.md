# library/methods/

音訊前處理方法（實驗變因）。所有方法都繼承 AudioMethod，介面為 process(wav, sr) → wav。

| 檔案 | 做什麼 |
|---|---|
| base.py | AudioMethod 抽象類別：load() / process() / unload() |
| registry.py | get_method()、list_methods()、register_method() |
| chain.py | MethodChain：依序串接，支援 "vad+denoise" |
| identity.py | baseline：原樣回傳 |
| vad.py | Silero VAD：保留語音段；完全無語音時回傳空陣列 |
| denoise.py | DeepFilterNet 降噪 |
| loudnorm.py | 響度正規化到目標 LUFS |

## 範例
    m = get_method("vad+denoise"); m.load("cpu")
    out = m.process(wav, 16000)

## 新增方法
1. 新增 xxx.py，繼承 AudioMethod 實作 process()
2. 在 registry.py 註冊名稱
3. 新增 configs/methods/xxx.yaml
4. 更新本 README
