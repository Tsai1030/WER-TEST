"""library.scoring.stats：配對 bootstrap 信賴區間、變好/變差統計。"""

import numpy as np
import pytest

from library.scoring import compare_per_utterance, paired_bootstrap


def test_bootstrap_deterministic() -> None:
    base, method, words = [3, 1, 4, 1, 5], [2, 1, 3, 2, 4], [10, 8, 12, 9, 11]
    assert paired_bootstrap(base, method, words, seed=1) == paired_bootstrap(base, method, words, seed=1)


def test_bootstrap_identical_errors_gives_zero_interval() -> None:
    assert paired_bootstrap([2, 3, 1], [2, 3, 1], [10, 10, 10]) == (0.0, 0.0)


def test_bootstrap_clear_improvement_is_significant() -> None:
    rng = np.random.default_rng(0)
    words = rng.integers(5, 15, size=50).tolist()
    base = [int(w * 0.3) for w in words]
    method = [max(b - 1, 0) for b in base]  # 每句少 1 個錯誤
    lo, hi = paired_bootstrap(base, method, words)
    observed = (sum(method) - sum(base)) / sum(words) * 100
    assert lo <= observed <= hi
    assert hi < 0  # plan.md 第 9 章：CI 上界 < 0 → 顯著改善


def test_bootstrap_noisy_difference_not_significant() -> None:
    base = [2, 0, 3, 1, 2, 0, 1, 3] * 5
    method = [1, 1, 3, 2, 1, 0, 2, 3] * 5  # 有好有壞，總和相同
    lo, hi = paired_bootstrap(base, method, [10] * 40)
    assert lo < 0 < hi


def test_bootstrap_input_errors() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1], [5, 5])
    with pytest.raises(ValueError):
        paired_bootstrap([], [], [])
    with pytest.raises(ValueError):
        paired_bootstrap([1], [0], [0])


def test_compare_per_utterance() -> None:
    assert compare_per_utterance([3, 2, 1, 0], [1, 2, 4, 0]) == {"improved": 1, "worse": 1, "same": 2}
    with pytest.raises(ValueError):
        compare_per_utterance([1], [1, 2])
