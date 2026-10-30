"""library.monitor：ResourceMonitor 峰值 RAM、CPU only 時 GPU 為 None、Timer。"""

import time

import numpy as np
import pytest

from library.monitor import ResourceMonitor, Timer


def test_resource_monitor_captures_ram_peak() -> None:
    with ResourceMonitor(poll_ms=10) as mon:
        block = np.ones(200 * 2**20 // 8)  # 約 200 MB
        time.sleep(0.2)
        del block
    assert mon.peak_ram_mb - mon.baseline_ram_mb > 150
    assert mon.elapsed_ms >= 200


def test_resource_monitor_cpu_only() -> None:
    import torch

    with ResourceMonitor() as mon:
        pass
    if torch.cuda.is_available():
        assert mon.gpu_measure == "torch" and mon.peak_gpu_mb is not None
    else:
        assert mon.gpu_measure == "none" and mon.peak_gpu_mb is None


def test_resource_monitor_explicit_none_and_invalid() -> None:
    with ResourceMonitor(gpu_measure="none") as mon:
        pass
    assert mon.peak_gpu_mb is None and mon.gpu_measure == "none"
    with pytest.raises(ValueError):
        ResourceMonitor(gpu_measure="nvidia-smi")


def test_timer() -> None:
    with Timer() as t:
        time.sleep(0.05)
    assert 45 <= t.ms < 1000
