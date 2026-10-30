"""英文正規化（Whisper EnglishTextNormalizer + 贅詞移除）。"""

import re
from functools import lru_cache

# Whisper normalizer 已移除 hmm、mm、mhm、mmm、uh、um；這裡補上其他常見拼法。
# 不含 ah、er、oh 等可能是一般詞彙的字。
EXTRA_FILLERS: tuple[str, ...] = ("uhm", "umm", "uhh", "erm", "hm")


@lru_cache(maxsize=1)
def _whisper_normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


@lru_cache(maxsize=8)
def _filler_pattern(fillers: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(map(re.escape, fillers)) + r")\b")


def normalize_english(text: str, extra_fillers: tuple[str, ...] = EXTRA_FILLERS) -> str:
    """英文正規化：小寫、去標點、縮寫展開、數字統一、英美拼法統一、移除贅詞（uh, um…）。

    先套用 Whisper EnglishTextNormalizer（例："two hundred" → "200"、"colour" → "color"、
    "don't" → "do not"），再移除 extra_fillers，最後合併多餘空白。

    參數:
        text: 原始文字（ref 或 hyp）。
        extra_fillers: Whisper normalizer 之外要移除的贅詞（需為小寫）。

    回傳:
        正規化後文字，詞與詞之間以單一空白分隔；空字串輸入回傳空字串。
    """
    out = _whisper_normalizer()(text)
    if extra_fillers:
        out = _filler_pattern(tuple(extra_fillers)).sub(" ", out)
    return " ".join(out.split())
