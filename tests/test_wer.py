"""library.scoring.wer：corpus_wer 為加權而非平均；與 jiwer 一致。"""

import random

import jiwer
import pytest

from library.scoring import align, corpus_wer


def test_corpus_wer_is_weighted_not_mean() -> None:
    results = [align("a", "b"), align("a b c d e f g h i j", "a b c d e f g h i j")]
    # 逐句 WER 平均 = (1.0 + 0.0) / 2 = 0.5；加權 = 1 / 11
    assert corpus_wer(results) == pytest.approx(1 / 11)


def test_corpus_wer_skips_noise_only_denominator() -> None:
    results = [align("a b", "a x"), align("", "hallucinated words")]
    assert corpus_wer(results) == pytest.approx((1 + 2) / 2)


def test_corpus_wer_requires_words() -> None:
    with pytest.raises(ValueError):
        corpus_wer([align("", "x")])
    with pytest.raises(ValueError):
        corpus_wer([])


def _random_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    vocab = ["a", "b", "c", "d", "e", "f", "g", "the", "cat", "sat", "on", "mat"]
    pairs = []
    for _ in range(n):
        ref = [rng.choice(vocab) for _ in range(rng.randint(1, 12))]
        hyp = list(ref)
        for _ in range(rng.randint(0, 5)):
            op = rng.choice(["sub", "del", "ins"])
            if op == "sub" and hyp:
                hyp[rng.randrange(len(hyp))] = rng.choice(vocab)
            elif op == "del" and hyp:
                hyp.pop(rng.randrange(len(hyp)))
            else:
                hyp.insert(rng.randint(0, len(hyp)), rng.choice(vocab))
        pairs.append((" ".join(ref), " ".join(hyp)))
    return pairs


def test_matches_jiwer_on_random_pairs() -> None:
    pairs = [(r, h) for r, h in _random_pairs(300, seed=0) if h]  # 排除空 hyp：避開 jiwer 對空字串的特殊處理
    refs, hyps = zip(*pairs, strict=True)
    ours = [align(r, h) for r, h in pairs]
    assert corpus_wer(ours) == pytest.approx(jiwer.wer(list(refs), list(hyps)))
    for (r, h), a in zip(pairs, ours, strict=True):
        out = jiwer.process_words(r, h)
        assert a.errors == out.substitutions + out.deletions + out.insertions
        assert a.N == out.substitutions + out.deletions + out.hits


@pytest.mark.parametrize(
    ("ref", "hyp"),
    [("i love you", "i love u"), ("a b c d", "a c d"), ("a c d", "a b c d"), ("a b c", "x y z")],
)
def test_sdi_matches_jiwer_on_unambiguous_cases(ref: str, hyp: str) -> None:
    a, out = align(ref, hyp), jiwer.process_words(ref, hyp)
    assert (a.S, a.D, a.I) == (out.substitutions, out.deletions, out.insertions)
