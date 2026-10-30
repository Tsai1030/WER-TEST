"""延遲計時。"""

from typing import Self


class Timer:
    """with Timer() as t: ... ; t.ms"""

    ms: float

    def __enter__(self) -> Self:
        """開始計時（time.perf_counter）。"""
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        """停止計時並將經過時間（毫秒）寫入 ms；不吞掉例外。"""
        raise NotImplementedError
