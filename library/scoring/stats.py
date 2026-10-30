"""配對 bootstrap 信賴區間、變好/變差統計。"""


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
        ValueError: 三個清單長度不一致或為空。
    """
    raise NotImplementedError


def compare_per_utterance(base_errors: list[int], method_errors: list[int]) -> dict:
    """回傳 {'improved': n, 'worse': n, 'same': n}。

    參數:
        base_errors: baseline 每句錯誤數。
        method_errors: 方法每句錯誤數，順序需與 base_errors 對應同一句。

    回傳:
        各類句數；improved 代表方法錯誤數較少。

    例外:
        ValueError: 兩個清單長度不一致。
    """
    raise NotImplementedError
