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
    def errors(self) -> int:
        """錯誤數 S + D + I。"""
        return self.S + self.D + self.I

    @property
    def wer(self) -> float | None:
        """(S + D + I) ÷ N；N = 0 時回傳 None。"""
        return self.errors / self.N if self.N else None


def align(ref: str, hyp: str) -> AlignmentResult:
    """輸入應為正規化後的文字。

    以空白切詞後做 Levenshtein 對齊（替換、刪除、插入成本皆為 1）。
    總錯誤數一定是最小編輯距離；同成本的多種對齊中，回溯時依序偏好 eq/sub、del、ins。

    參數:
        ref: 正規化後 reference。
        hyp: 正規化後 hypothesis。

    回傳:
        AlignmentResult；ref 為空時 N = 0、I = hyp 詞數。
    """
    r, h = ref.split(), hyp.split()
    n, m = len(r), len(h)
    # cost[i][j]：r[:i] 與 h[:j] 的最小編輯距離
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = cost[i - 1][j - 1] + (r[i - 1] != h[j - 1])
            cost[i][j] = min(diag, cost[i - 1][j] + 1, cost[i][j - 1] + 1)

    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("eq" if r[i - 1] == h[j - 1] else "sub", r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", r[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", h[j - 1]))
            j -= 1
    ops.reverse()

    counts = {op: 0 for op in ("sub", "del", "ins")}
    for op, _, _ in ops:
        if op in counts:
            counts[op] += 1
    return AlignmentResult(S=counts["sub"], D=counts["del"], I=counts["ins"], N=n, ops=ops)
