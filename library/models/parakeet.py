"""Parakeet TDT 0.6B v2（暫緩：保留空殼，不實作）。

依賴 NeMo，官方以 Linux 為主，Windows 原生安裝困難；
待有 Linux 環境（WSL2 或 Jetson）再實作，並將 configs/models/parakeet-tdt-v2.yaml 改為 enabled: true。
"""

import numpy as np

from library.models.base import ASRModel, Transcript


class ParakeetModel(ASRModel):
    """Parakeet TDT 0.6B v2：FastConformer + TDT（Transducer），greedy 解碼。暫緩。"""

    name = "parakeet-tdt-v2"
    framework = "nemo"

    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        """暫緩：需 Linux 環境與 nemo_toolkit。

        例外:
            NotImplementedError: 本輪不實作。
        """
        raise NotImplementedError("parakeet-tdt-v2 暫緩：需 Linux 環境（NeMo）")

    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript:
        """暫緩。

        例外:
            NotImplementedError: 本輪不實作。
        """
        raise NotImplementedError("parakeet-tdt-v2 暫緩：需 Linux 環境（NeMo）")
