# library/models/

STT 模型包裝。所有模型繼承 ASRModel，介面為 transcribe(wav, sr) → Transcript。

| 檔案 | 模型 | 架構 |
|---|---|---|
| base.py | ASRModel 抽象類別、Transcript 資料結構 | — |
| registry.py | get_model()、list_models()、register_model() | — |
| granite.py | Granite Speech 5.0 470M TurboCTC | CTC |
| parakeet.py | Parakeet TDT 0.6B v2（暫緩，需 Linux／NeMo，目前只有空殼） | Transducer |
| whisper.py | Whisper small（transformers；可切換 faster-whisper） | Encoder-decoder |
| qwen3_asr.py | Qwen3-ASR-0.6B（transformers 原生版 `-hf`） | Audio-LLM |

## 範例
    model = get_model("granite-470m"); model.load("cpu", "fp32")
    print(model.transcribe(wav, 16000).text)
    model.unload()

## 新增模型
1. 新增 xxx.py，繼承 ASRModel，實作 load() 與 transcribe()
2. describe() 回傳框架與解碼設定（寫入 meta.json）
3. 在 registry.py 註冊；新增 configs/models/xxx.yaml；更新本 README

## 注意
- transcribe() 不做正規化，回傳模型原始輸出
- unload() 必須釋放記憶體，否則下一個模型的峰值會被污染
