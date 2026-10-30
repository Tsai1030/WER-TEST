"""AudioMethod 抽象類別。"""

from abc import ABC, abstractmethod

import numpy as np


class AudioMethod(ABC):
    """音訊前處理方法的共同介面。

    子類別需設定 name（見 plan.md 6.5）並實作 process()；
    需要載入模型的方法（VAD、降噪）在 load() 內才 import 重量級套件。
    """

    name: str

    def __init__(self, config: dict | None = None) -> None:
        """參數:
            config: configs/methods/<name>.yaml 讀入後的 dict；None 表示使用預設值。
        """
        raise NotImplementedError

    def load(self, device: str = "cpu") -> None:
        """載入模型（如 VAD、降噪模型）；不需模型的方法可不覆寫。

        參數:
            device: 執行裝置；cuda 不可用時退回 cpu 並記錄警告。
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """處理一段音訊。

        參數:
            wav: 一維 float32 陣列。
            sr: 取樣率。

        回傳:
            處理後的音訊（取樣率不變）。可能為長度 0 的陣列（VAD 判斷無語音）。

        例外:
            RuntimeError: 需要模型的方法尚未呼叫 load()。
        """
        raise NotImplementedError

    def unload(self) -> None:
        """釋放 load() 載入的資源。"""
        raise NotImplementedError
