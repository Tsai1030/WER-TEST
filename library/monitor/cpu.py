"""行程 CPU 時間與目前記憶體量測。"""

import time
from typing import Self

import psutil


def current_rss_mb() -> float:
    """回傳目前行程的 RSS（MB）。"""
    return psutil.Process().memory_info().rss / 2**20


class CpuTimer:
    """with CpuTimer() as t: ... ; t.cpu_sec（行程 CPU 秒數，所有執行緒合計）、t.wall_sec（經過秒數）。

    cpu_sec ÷ 音訊秒數 × 100 = 即時處理時佔用一顆邏輯核心的百分比（大於 100 代表用到多顆）。
    Windows 的 CPU 時間解析度約 16 ms，量測區間需數秒以上。
    """

    cpu_sec: float
    wall_sec: float

    def __init__(self) -> None:
        self._process = psutil.Process()

    def _cpu(self) -> float:
        t = self._process.cpu_times()
        return t.user + t.system

    def __enter__(self) -> Self:
        """記錄起點。"""
        self.cpu_sec = self.wall_sec = 0.0
        self._cpu0, self._wall0 = self._cpu(), time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        """記錄 cpu_sec、wall_sec；不吞掉例外。"""
        self.cpu_sec = self._cpu() - self._cpu0
        self.wall_sec = time.perf_counter() - self._wall0
