"""ASRModel 抽象類別、Transcript 資料結構。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Transcript:
    """一次辨識的結果。"""

    text: str  # 模型原始輸出文字
    language: str | None = None  # 多語模型偵測到的語言
    extra: dict | None = None  # 模型特有資訊（例如 Whisper 的 no_speech_prob）


class ASRModel(ABC):
    """STT 模型的共同介面。

    子類別需設定 name（見 plan.md 6.5）與 framework，並實作 load() 與 transcribe()。
    重量級套件（torch、transformers…）在 load() 內才 import。
    """

    name: str
    framework: str

    def __init__(self, config: dict) -> None:
        """參數:
            config: configs/models/<name>.yaml 讀入後的 dict（hf_id、decoding…）。
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        """載入模型權重。

        參數:
            device: 執行裝置；cuda 不可用時退回 cpu 並記錄警告。
            dtype: 精度（CPU 用 fp32；Jetson 用 fp16）。
            num_threads: CPU 執行緒數（torch.set_num_threads）。

        例外:
            ImportError: 模型所需套件未安裝。
            ValueError: dtype 不支援。
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript:
        """辨識一段音訊；不做正規化，回傳模型原始輸出。

        參數:
            wav: 一維 float32 陣列（長度需 > 0，空音訊由 runner 處理）。
            sr: 取樣率。

        回傳:
            Transcript。

        例外:
            RuntimeError: 尚未呼叫 load()。
        """
        raise NotImplementedError

    def unload(self) -> None:
        """釋放記憶體（del 模型 + gc.collect；有 GPU 時另呼叫 torch.cuda.empty_cache）。"""
        raise NotImplementedError

    def describe(self) -> dict:
        """回傳 framework、hf_id、decoding 等資訊，寫入 meta.json。"""
        raise NotImplementedError
