"""展開「模型 × 方法 × 條件 × 句子」組合。"""

from collections.abc import Iterator

from library.data import Utterance
from library.runner.config import ExperimentConfig


def iter_matrix(cfg: ExperimentConfig) -> Iterator[tuple[str, str, str, Utterance]]:
    """依「模型 → 方法 → 條件 → 句子」順序展開（最外層為模型，模型只載入一次）。

    noise_only 條件使用純噪音片段（noise01…）而非 manifest 中的語音句。

    參數:
        cfg: 實驗設定。

    回傳:
        依序產生 (model, method, condition, utterance)。

    例外:
        FileNotFoundError: manifest 或劣化音檔資料夾不存在。
    """
    raise NotImplementedError
