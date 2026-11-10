# library/

核心函式庫。每個子模組只負責一件事，彼此不互相依賴，只有 runner/ 負責串接。

| 子模組 | 職責 | 主要入口 |
|---|---|---|
| data/ | 讀資料集、抽樣、轉檔、manifest | sample_utterances, build_manifest, load_manifest |
| degrade/ | 產生劣化音訊 | apply_condition, add_noise |
| methods/ | 音訊前處理方法 | get_method("vad+denoise") |
| models/ | STT 模型統一介面 | get_model("whisper-small") |
| textnorm/ | 文字正規化 | normalize |
| scoring/ | 對齊、WER、統計 | align, corpus_wer, paired_bootstrap |
| monitor/ | 記憶體、延遲量測 | ResourceMonitor, Timer |
| kws/ | 喚醒詞偵測（sherpa-onnx） | KeywordSpotter |
| endpoint/ | 講完判斷（串流 VAD） | Endpointer, decide |
| llm/ | LLM client（連到另外啟動的 llama-server，串流） | LLMClient |
| tts/ | TTS client（連到另外啟動的 services/pocket_tts_server 或 tts_server，支援串流） | TTSClient |
| runner/ | 實驗矩陣執行；Round 4 喚醒測試與即時實測；整段語音助理實測 | run_experiment, run_kws_negative, LiveDemo, LiveAssistant |
| export/ | 輸出結果檔 | ResultWriter, validate_results |
| utils/ | 路徑、log、seed | — |

## 高階 API（__init__.py）
- transcribe_file(wav_path, model, method) → str
- evaluate(ref, hyp) → AlignmentResult
- run_experiment(config_path) → Path
