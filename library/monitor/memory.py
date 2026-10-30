"""峰值 RAM（psutil）；有 GPU 時另量 GPU。"""

from typing import Self


class ResourceMonitor:
    """Context manager。進入時記錄基準，離開時記錄峰值。
    RAM（主要指標）：psutil 背景執行緒輪詢行程 RSS（每 50ms）。
    GPU（選用）：偵測到 CUDA 時才量；PyTorch 用 torch.cuda.max_memory_allocated，
         其他框架用 pynvml 輪詢。無 GPU 時 peak_gpu_mb = None。"""

    peak_gpu_mb: float | None
    peak_ram_mb: float
    elapsed_ms: float

    def __init__(self, poll_ms: int = 50, gpu_measure: str = "auto") -> None:
        """參數:
            poll_ms: RSS 輪詢間隔（毫秒）。
            gpu_measure: "auto"（偵測 CUDA）、"none"、"torch" 或 "pynvml"；
                實際採用的方式需寫入 resources.csv 的 gpu_measure 欄。
        """
        raise NotImplementedError

    def __enter__(self) -> Self:
        """記錄基準並啟動背景輪詢執行緒。"""
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        """停止輪詢並記錄 peak_ram_mb、peak_gpu_mb、elapsed_ms；不吞掉例外。"""
        raise NotImplementedError
