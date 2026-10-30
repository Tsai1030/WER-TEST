"""M0 速度實測：單一模型辨識少量 clean 音檔，估算完整矩陣所需時間。"""

import os
from dataclasses import dataclass, field

import psutil

from library.data import Utterance, read_audio
from library.models import get_model
from library.monitor import Timer


@dataclass
class BenchResult:
    """單一模型的實測結果。"""

    model: str
    load_s: float
    rss_after_load_mb: float
    warmup_ms: float
    utt_ms: list[float] = field(default_factory=list)
    audio_sec: list[float] = field(default_factory=list)
    samples: list[tuple[str, str, str]] = field(default_factory=list)  # (utt_id, ref_raw, hyp_raw)

    @property
    def mean_sec(self) -> float:
        """平均每句秒數（不含暖機）。"""
        return sum(self.utt_ms) / len(self.utt_ms) / 1000.0

    @property
    def rtf(self) -> float:
        """處理時間 ÷ 音檔長度。"""
        return sum(self.utt_ms) / 1000.0 / sum(self.audio_sec)


def inferences_per_model(n_methods: int, conditions: list[str], n_speech: int, n_noise: int) -> int:
    """每個模型在完整矩陣中的推論次數：方法數 ×（語音條件數 × 語音句數 + 純噪音片段數）。

    參數:
        n_methods: 方法數（含 baseline 與組合）。
        conditions: 條件名稱清單（noise_only 以 n_noise 計）。
        n_speech: 語音句數。
        n_noise: 純噪音片段數。

    回傳:
        推論次數。Round 1：4 ×（7 × 50 + 10）= 1,440。
    """
    n_speech_conds = sum(c != "noise_only" for c in conditions)
    n_noise_conds = len(conditions) - n_speech_conds
    return n_methods * (n_speech_conds * n_speech + n_noise_conds * n_noise)


def bench_model(
    name: str,
    utts: list[Utterance],
    device: str = "cpu",
    dtype: str = "fp32",
    num_threads: int = 4,
    config: dict | None = None,
) -> BenchResult:
    """載入模型、以第一句暖機（不計入），再逐句計時其餘音檔，結束後 unload。

    參數:
        name: 模型名稱。
        utts: 測試音檔；第一句只用於暖機，其餘計時（至少 2 句）。
        device: 執行裝置。
        dtype: 精度。
        num_threads: CPU 執行緒數。
        config: 模型設定 dict；None 表示讀取 configs/models/<name>.yaml。

    回傳:
        BenchResult。

    例外:
        ValueError: utts 少於 2 句。
    """
    if len(utts) < 2:
        raise ValueError("utts 至少需要 2 句（第一句暖機）")
    model = get_model(name, config)
    try:
        with Timer() as t_load:
            model.load(device, dtype, num_threads)
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 2**20

        warm_wav, warm_sr = read_audio(utts[0].wav_path)
        with Timer() as t_warm:
            model.transcribe(warm_wav, warm_sr)

        result = BenchResult(name, t_load.ms / 1000.0, rss_mb, t_warm.ms)
        for u in utts[1:]:
            wav, sr = read_audio(u.wav_path)
            with Timer() as t:
                hyp = model.transcribe(wav, sr).text
            result.utt_ms.append(t.ms)
            result.audio_sec.append(u.audio_sec)
            result.samples.append((u.utt_id, u.ref_raw, hyp))
        return result
    finally:
        model.unload()
