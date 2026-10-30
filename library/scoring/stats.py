"""配對 bootstrap 信賴區間、變好/變差統計。"""

import numpy as np


def _check_lengths(*arrays: list[int]) -> None:
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"清單長度不一致：{[len(a) for a in arrays]}")
    if lengths == {0}:
        raise ValueError("清單不可為空")


def paired_bootstrap(
    base_errors: list[int],
    method_errors: list[int],
    n_words: list[int],
    n_boot: int = 1000,
    seed: int = 12345,
) -> tuple[float, float]:
    """回傳 ΔWER（百分點）的 95% 信賴區間。

    以句為單位配對重抽 n_boot 次，每次計算 (Σmethod − Σbase) ÷ ΣN × 100，
    取 2.5% 與 97.5% 分位數。

    參數:
        base_errors: baseline 每句錯誤數（S+D+I）。
        method_errors: 方法每句錯誤數，順序需與 base_errors 對應同一句。
        n_words: 每句 reference 詞數 N。
        n_boot: 重抽次數。
        seed: 亂數種子。

    回傳:
        (下界, 上界)，單位為百分點；負值代表改善。

    例外:
        ValueError: 三個清單長度不一致或為空，或 ΣN = 0。
    """
    _check_lengths(base_errors, method_errors, n_words)
    diff = np.asarray(method_errors, dtype=np.float64) - np.asarray(base_errors, dtype=np.float64)
    words = np.asarray(n_words, dtype=np.float64)
    if words.sum() == 0:
        raise ValueError("ΣN = 0，無法計算 ΔWER")
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(n_boot, len(diff)))
    sum_words = words[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        deltas = diff[idx].sum(axis=1) / sum_words * 100
    deltas = deltas[sum_words > 0]  # 重抽到全是 N=0 的句子時略過
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def compare_per_utterance(base_errors: list[int], method_errors: list[int]) -> dict:
    """回傳 {'improved': n, 'worse': n, 'same': n}。

    參數:
        base_errors: baseline 每句錯誤數。
        method_errors: 方法每句錯誤數，順序需與 base_errors 對應同一句。

    回傳:
        各類句數；improved 代表方法錯誤數較少。

    例外:
        ValueError: 兩個清單長度不一致或為空。
    """
    _check_lengths(base_errors, method_errors)
    diff = np.asarray(method_errors) - np.asarray(base_errors)
    return {"improved": int((diff < 0).sum()), "worse": int((diff > 0).sum()), "same": int((diff == 0).sum())}
