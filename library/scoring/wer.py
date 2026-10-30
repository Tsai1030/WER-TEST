"""逐句 / 整體 WER。逐句 WER 見 AlignmentResult.wer。"""

from collections.abc import Iterable

from library.scoring.align import AlignmentResult


def corpus_wer(results: Iterable[AlignmentResult]) -> float:
    """整體 WER = ΣS+D+I ÷ ΣN（不是逐句 WER 的平均）。

    參數:
        results: 各句對齊結果。

    回傳:
        整體 WER（0 以上，可大於 1）。

    例外:
        ValueError: ΣN = 0（例如只有純噪音片段）。
    """
    raise NotImplementedError
