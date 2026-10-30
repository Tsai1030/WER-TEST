"""峰值 RAM（psutil）；有 GPU 時另量 GPU。"""

import os
import sys
import threading
import time
from typing import Self

import psutil

_GPU_MEASURES = ("auto", "none", "torch", "pynvml")


def _cuda_available() -> bool:
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch
        except ImportError:
            return False
    return bool(torch.cuda.is_available())


class ResourceMonitor:
    """Context manager。進入時記錄基準，離開時記錄峰值。
    RAM（主要指標）：psutil 背景執行緒輪詢行程 RSS（每 50ms）。
    GPU（選用）：偵測到 CUDA 時才量；PyTorch 用 torch.cuda.max_memory_allocated，
         其他框架用 pynvml 輪詢。無 GPU 時 peak_gpu_mb = None。"""

    peak_gpu_mb: float | None
    peak_ram_mb: float
    baseline_ram_mb: float
    elapsed_ms: float
    gpu_measure: str

    def __init__(self, poll_ms: int = 50, gpu_measure: str = "auto") -> None:
        """參數:
            poll_ms: RSS 輪詢間隔（毫秒）。
            gpu_measure: "auto"（偵測 CUDA，有則用 torch）、"none"、"torch" 或 "pynvml"；
                實際採用的方式記在 self.gpu_measure，需寫入 resources.csv 的 gpu_measure 欄。

        例外:
            ValueError: gpu_measure 不合法。
        """
        if gpu_measure not in _GPU_MEASURES:
            raise ValueError(f"gpu_measure 必須是 {_GPU_MEASURES} 之一：{gpu_measure}")
        self._poll_s = poll_ms / 1000.0
        self._requested = gpu_measure
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_handle = None
        self.peak_gpu_mb = None
        self.peak_ram_mb = 0.0
        self.baseline_ram_mb = 0.0
        self.elapsed_ms = 0.0
        self.gpu_measure = "none"

    def _rss_mb(self) -> float:
        return self._process.memory_info().rss / 2**20

    def _poll(self) -> None:
        while not self._stop.wait(self._poll_s):
            self.peak_ram_mb = max(self.peak_ram_mb, self._rss_mb())
            if self.gpu_measure == "pynvml":
                import pynvml

                used = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used / 2**20
                self.peak_gpu_mb = max(self.peak_gpu_mb or 0.0, used)

    def _start_gpu(self) -> None:
        mode = self._requested
        if mode == "auto":
            mode = "torch" if _cuda_available() else "none"
        if mode == "torch":
            import torch

            torch.cuda.reset_peak_memory_stats()
            self.peak_gpu_mb = 0.0
        elif mode == "pynvml":
            import pynvml

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.peak_gpu_mb = 0.0
        self.gpu_measure = mode

    def __enter__(self) -> Self:
        """記錄基準並啟動背景輪詢執行緒。"""
        self._start_gpu()
        self.baseline_ram_mb = self.peak_ram_mb = self._rss_mb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._start = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        """停止輪詢並記錄 peak_ram_mb、peak_gpu_mb、elapsed_ms；不吞掉例外。"""
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_ram_mb = max(self.peak_ram_mb, self._rss_mb())
        if self.gpu_measure == "torch":
            import torch

            self.peak_gpu_mb = torch.cuda.max_memory_allocated() / 2**20
        elif self.gpu_measure == "pynvml":
            import pynvml

            pynvml.nvmlShutdown()
