"""STT WER 測試專案核心函式庫：對外的高階 API。

給「只想快速用」的人；細部控制請直接使用各子模組（見 library/README.md）。
"""

from pathlib import Path

from library.scoring import AlignmentResult

__all__ = ["AlignmentResult", "evaluate", "run_experiment", "transcribe_file"]


def transcribe_file(
    wav_path: str | Path,
    model: str = "whisper-small",
    method: str = "baseline",
    device: str = "cpu",
) -> str:
    """讀取音檔 → 套用前處理方法 → 辨識，回傳文字。

    參數:
        wav_path: 音檔路徑。
        model: 模型名稱（見 plan.md 6.5，例：whisper-small）。
        method: 前處理方法名稱，可用 '+' 串接（例：vad+denoise）。
        device: 執行裝置；cuda 不可用時退回 cpu。

    回傳:
        模型原始輸出文字（未正規化）。

    例外:
        FileNotFoundError: 音檔不存在。
        KeyError: 模型或方法名稱未註冊。
    """
    from library.data import read_audio
    from library.methods import get_method
    from library.models import get_model

    wav, sr = read_audio(wav_path)
    if sr != 16000:
        import librosa

        wav, sr = librosa.resample(wav, orig_sr=sr, target_sr=16000), 16000
    m = get_method(method)
    m.load(device)
    try:
        processed = m.process(wav, sr)
    finally:
        m.unload()
    if len(processed) == 0:  # VAD 判斷無語音
        return ""
    asr = get_model(model)
    asr.load(device)
    try:
        return asr.transcribe(processed, sr).text
    finally:
        asr.unload()


def evaluate(ref: str, hyp: str, normalizer: str = "english") -> AlignmentResult:
    """正規化後對齊 ref 與 hyp，回傳 WER 與 S/D/I。

    參數:
        ref: reference 原始文字。
        hyp: 模型輸出原始文字。
        normalizer: 正規化器名稱（見 library.textnorm.get_normalizer）。

    回傳:
        AlignmentResult（含 S、D、I、N、ops 與 wer）。

    例外:
        KeyError: normalizer 名稱未註冊。
    """
    from library.scoring import align
    from library.textnorm import get_normalizer

    norm = get_normalizer(normalizer)
    return align(norm(ref), norm(hyp))


def run_experiment(config_path: str | Path) -> Path:
    """依 YAML 設定跑完整實驗矩陣，回傳結果資料夾路徑。

    參數:
        config_path: 實驗設定檔路徑（例：configs/round1.yaml）。

    回傳:
        結果資料夾路徑（含 results.csv、resources.csv、meta.json）。

    例外:
        FileNotFoundError: 設定檔不存在。
        ValueError: 設定檔欄位不合法。
    """
    from library import runner

    return runner.run_experiment(runner.load_config(config_path))
