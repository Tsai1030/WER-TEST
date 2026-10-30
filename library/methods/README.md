# library/methods/

音訊前處理方法（實驗變因）。所有方法都繼承 AudioMethod，介面為 process(wav, sr) → wav。

| 檔案 | 做什麼 |
|---|---|
| base.py | AudioMethod 抽象類別：load() / process() / unload() / describe() |
| registry.py | get_method()、list_methods()、register_method()；未給設定時讀 configs/methods/<name>.yaml |
| chain.py | MethodChain：依序串接，支援 "vad+denoise"；中途變成空音訊即停止 |
| identity.py | baseline：原樣回傳 |
| vad.py | Silero VAD：保留語音段並串接；完全無語音時回傳空陣列（權重內建於套件，CPU 執行） |
| denoise.py | 降噪（noisereduce 非穩態頻譜閘控；原規劃 DeepFilterNet，因與新版 torchaudio 不相容而停用） |
| loudnorm.py | 響度正規化到目標 LUFS（pyloudnorm）；短於 0.4 秒或整段靜音時原樣回傳 |

## 範例
    m = get_method("vad+denoise"); m.load("cpu")
    out = m.process(wav, 16000)

## 新增方法
1. 新增 xxx.py，繼承 AudioMethod 實作 process()
2. 在 registry.py 註冊名稱
3. 新增 configs/methods/xxx.yaml
4. 更新本 README
