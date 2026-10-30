"""對齊與評分。"""

from library.scoring.align import AlignmentResult, align
from library.scoring.stats import compare_per_utterance, paired_bootstrap
from library.scoring.wer import corpus_wer

__all__ = [
    "AlignmentResult",
    "align",
    "compare_per_utterance",
    "corpus_wer",
    "paired_bootstrap",
]
