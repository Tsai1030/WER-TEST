"""三個輸出檔的欄位定義（與 dashboard/data_contract.md 一致，不可隨意更改）。"""

from dataclasses import dataclass, fields

# plan.md 6.5 命名規則：網頁依此自動分組
MODEL_NAMES: tuple[str, ...] = ("granite-470m", "whisper-small", "qwen3-asr-0.6b", "parakeet-tdt-v2")
CONDITION_NAMES: tuple[str, ...] = (
    "clean",
    "snr20",
    "snr10",
    "snr05",
    "snr00",
    "reverb",
    "lowvol",
    "noise_only",
)
METHOD_NAMES: tuple[str, ...] = ("baseline", "vad", "denoise", "loudnorm")  # 組合以 '+' 連接


@dataclass
class ResultRow:
    """results.csv 一列（每列一個「模型 × 條件 × 方法 × 句子」，見 plan.md 6.2）。"""

    model: str
    condition: str
    method: str
    utt_id: str
    speaker_id: str
    gender: str
    audio_sec: float  # 原始音檔秒數（計算 RTF 用）
    ref_raw: str  # 純噪音為空
    hyp_raw: str  # 模型原始輸出
    ref_norm: str
    hyp_norm: str
    wer: float | None  # 0–1；純噪音為空
    S: int  # 純噪音：S=D=0、I=hyp 詞數、N=0
    D: int
    I: int
    N: int
    latency_ms: float  # 方法處理 + 模型辨識的總時間
    method_ms: float  # 僅方法處理時間
    detected_lang: str  # 多語模型偵測語言，其他留空


@dataclass
class ResourceRow:
    """resources.csv 一列（每列一個「模型 × 方法」，見 plan.md 6.3）。"""

    model: str
    method: str
    peak_gpu_mb: float | None  # CPU only 時留空
    peak_ram_mb: float  # 本輪主要記憶體指標
    model_size_mb: float
    load_time_s: float
    gpu_measure: str  # none / torch / pynvml（不同測法不可直接比較）


RESULT_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(ResultRow))
RESOURCE_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(ResourceRow))
GPU_MEASURES: tuple[str, ...] = ("none", "torch", "pynvml")
META_REQUIRED_KEYS: tuple[str, ...] = (
    "round", "date", "git_commit", "device", "cpu", "ram_gb", "gpu", "num_threads", "env", "python",
    "precision", "dataset", "normalizer", "nano_budget_mb", "packages", "models", "methods", "conditions",
    "matrix",
)


def result_key(model: str, method: str, condition: str, utt_id: str) -> str:
    """results.csv 一列的唯一鍵（checkpoint 與驗證共用）。"""
    return f"{model}|{method}|{condition}|{utt_id}"
