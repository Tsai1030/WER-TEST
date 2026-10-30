"""Levenshtein 對齊，輸出 S/D/I 與對齊序列。"""

from dataclasses import dataclass


@dataclass
class AlignmentResult:
    """單句對齊結果。"""

    S: int
    D: int
    I: int
    N: int  # reference 詞數
    ops: list[tuple[str, str, str]]  # (op, ref_word, hyp_word)，op ∈ {"eq","sub","del","ins"}

    @property
    def wer(self) -> float | None:
        """(S + D + I) ÷ N；N = 0 時回傳 None。"""
        raise NotImplementedError


def align(ref: str, hyp: str) -> AlignmentResult:
    """輸入應為正規化後的文字。

    以空白切詞後做 Levenshtein 對齊（替換、刪除、插入成本皆為 1）。

    參數:
        ref: 正規化後 reference。
        hyp: 正規化後 hypothesis。

    回傳:
        AlignmentResult；ref 為空時 N = 0、I = hyp 詞數。
    """
    raise NotImplementedError
