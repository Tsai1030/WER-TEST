# library/

核心函式庫。每個子模組只負責一件事，彼此不互相依賴，只有 runner/ 負責串接。

| 子模組 | 職責 | 主要入口 |
|---|---|---|
| data/ | 讀資料集、抽樣、轉檔、manifest | sample_utterances, build_manifest, load_manifest |
| degrade/ | 產生劣化音訊 | apply_condition, add_noise |
| methods/ | 音訊前處理方法 | get_method("vad+denoise") |
| models/ | STT 模型統一介面 | get_model("whisper-turbo") |
| textnorm/ | 文字正規化 | normalize |
| scoring/ | 對齊、WER、統計 | align, corpus_wer, paired_bootstrap |
| monitor/ | 記憶體、延遲量測 | ResourceMonitor, Timer |
| runner/ | 實驗矩陣執行 | run_experiment |
| export/ | 輸出結果檔 | ResultWriter, validate_results |
| utils/ | 路徑、log、seed | — |

## 高階 API（__init__.py）
- transcribe_file(wav_path, model, method) → str
- evaluate(ref, hyp) → AlignmentResult
- run_experiment(config_path) → Path
