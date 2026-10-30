"""library.scoring.align：S/D/I 計算、空字串、全錯等邊界情況。"""

import pytest

from library import evaluate
from library.scoring import align


def _reconstruct(ops: list[tuple[str, str, str]]) -> tuple[str, str]:
    ref = " ".join(r for op, r, _ in ops if op != "ins")
    hyp = " ".join(h for op, _, h in ops if op != "del")
    return ref, hyp


def test_readme_example() -> None:
    r = align("i love you", "i love u")
    assert (r.S, r.D, r.I, r.N) == (1, 0, 0, 3)
    assert r.wer == pytest.approx(1 / 3)
    assert r.ops == [("eq", "i", "i"), ("eq", "love", "love"), ("sub", "you", "u")]


def test_identical() -> None:
    r = align("a b c", "a b c")
    assert (r.S, r.D, r.I, r.errors, r.wer) == (0, 0, 0, 0, 0.0)


def test_deletion_and_insertion() -> None:
    r = align("a b c d", "a c d")
    assert (r.S, r.D, r.I) == (0, 1, 0)
    assert ("del", "b", "") in r.ops
    r = align("a c d", "a b c d")
    assert (r.S, r.D, r.I) == (0, 0, 1)
    assert ("ins", "", "b") in r.ops


def test_empty_hyp_all_deletions() -> None:
    r = align("a b c", "")
    assert (r.S, r.D, r.I, r.N, r.wer) == (0, 3, 0, 3, 1.0)


def test_empty_ref_noise_only() -> None:
    """純噪音：S=D=0、I=hyp 詞數、N=0，wer 為 None。"""
    r = align("", "thank you for watching")
    assert (r.S, r.D, r.I, r.N) == (0, 0, 4, 0)
    assert r.wer is None


def test_both_empty() -> None:
    r = align("", "")
    assert (r.errors, r.N, r.ops, r.wer) == (0, 0, [], None)


def test_all_wrong_and_longer_hyp() -> None:
    r = align("a b", "x y z w")
    assert r.errors == 4 and r.N == 2 and r.wer == 2.0  # WER 可大於 1
    assert r.S == 2 and r.I == 2


def test_ops_reconstruct_inputs() -> None:
    ref, hyp = "the cat sat on the mat today", "a cat sat in on mat"
    r = align(ref, hyp)
    assert _reconstruct(r.ops) == (ref, hyp)
    assert r.errors == sum(op != "eq" for op, _, _ in r.ops)


def test_whitespace_insensitive() -> None:
    assert align("  a   b ", "a b").errors == 0


def test_evaluate_normalizes_both_sides() -> None:
    r = evaluate("I love you.", "i love u")  # plan.md 附錄 B.2
    assert (r.wer, r.S, r.D, r.I) == (pytest.approx(1 / 3), 1, 0, 0)
    assert evaluate("The file was two hundred bytes.", "the file was 200 bytes").errors == 0
