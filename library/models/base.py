"""ASRModel 抽象類別、Transcript 資料結構。"""

import gc
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TORCH_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


@dataclass
class Transcript:
    """一次辨識的結果。"""

    text: str  # 模型原始輸出文字
    language: str | None = None  # 多語模型自動偵測到的語言；設定檔強制語言時為 None
    extra: dict | None = None  # 模型特有資訊（例如 Whisper 的 no_speech_prob）


def torch_dtype(dtype: str) -> Any:
    """將設定檔的 dtype（fp32 / fp16 / bf16）轉為 torch.dtype。

    例外:
        ValueError: dtype 不支援。
    """
    if dtype not in _TORCH_DTYPES:
        raise ValueError(f"不支援的 dtype：{dtype}（可用：{list(_TORCH_DTYPES)}）")
    import torch

    return getattr(torch, _TORCH_DTYPES[dtype])


class ASRModel(ABC):
    """STT 模型的共同介面。

    子類別需設定 name（見 plan.md 6.5）與 framework，並實作 load() 與 transcribe()。
    重量級套件（torch、transformers…）在 load() 內才 import。
    子類別把載入的物件放在 self._model / self._processor，unload() 會統一釋放。
    """

    name: str
    framework: str
    supported_decoding: tuple[str, ...] = ()  # 可由解碼方法覆寫的參數（例：beam_size）

    def __init__(self, config: dict) -> None:
        """參數:
            config: configs/models/<name>.yaml 讀入後的 dict（hf_id、decoding…）。
        """
        self.config = dict(config)
        self.device = "cpu"
        self.dtype = "fp32"
        self._model: Any = None
        self._processor: Any = None
        self._decoding_overrides: dict = {}

    @property
    def hf_id(self) -> str:
        """設定檔中的 Hugging Face ID。"""
        return self.config["hf_id"]

    @property
    def revision(self) -> str | None:
        """設定檔中鎖定的 Hugging Face 版本（commit hash）；None 表示使用最新版。"""
        return self.config.get("revision")

    @property
    def decoding(self) -> dict:
        """實際使用的解碼參數：設定檔的 decoding，再套用 set_decoding() 的覆寫。"""
        return {**(self.config.get("decoding") or {}), **self._decoding_overrides}

    def set_decoding(self, overrides: dict) -> None:
        """設定解碼參數覆寫（由 runner 依方法設定；{} 表示回到設定檔預設）。

        參數:
            overrides: 例：{"beam_size": 5}。

        例外:
            ValueError: 含本模型不支援的參數（見 supported_decoding）。
        """
        bad = sorted(set(overrides) - set(self.supported_decoding))
        if bad:
            raise ValueError(f"{self.name} 不支援解碼參數 {bad}（可覆寫：{list(self.supported_decoding)}）")
        self._decoding_overrides = dict(overrides)

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(f"{self.name} 尚未呼叫 load()")

    @abstractmethod
    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None:
        """載入模型權重。

        參數:
            device: 執行裝置；cuda 不可用時退回 cpu 並記錄警告。
            dtype: 精度（開發機 CPU 用 fp32；Jetson 部署的 q4 版本另以部署引擎複驗）。
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
        self._model = None
        self._processor = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def weights_size_mb(self) -> float | None:
        """模型權重檔大小（MB），自本機 Hugging Face 快取計算；找不到時回傳 None。

        只計 *.safetensors、*.bin、*.pt，不含 tokenizer 與設定檔。不連網。
        """
        from huggingface_hub import snapshot_download

        info = self.describe()
        try:
            folder = snapshot_download(info["hf_id"], revision=info.get("revision"), local_files_only=True)
        except Exception:  # noqa: BLE001  快取不存在、離線等都視為無法取得
            return None
        files = [p for ext in ("*.safetensors", "*.bin", "*.pt") for p in Path(folder).rglob(ext)]
        return round(sum(p.stat().st_size for p in files) / 2**20, 1) if files else None

    def describe(self) -> dict:
        """回傳 framework、hf_id、decoding 等資訊，寫入 meta.json。"""
        return {
            "hf_id": self.config.get("hf_id"),
            "revision": self.revision,
            "framework": self.framework,
            "architecture": self.config.get("architecture"),
            "decoding": self.config.get("decoding"),
            "language": self.config.get("language"),
            "device": self.device,
            "dtype": self.dtype,
        }
