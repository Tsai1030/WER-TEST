"""英文正規化（Whisper EnglishTextNormalizer + 贅詞移除）。"""


def normalize_english(text: str) -> str:
    """英文正規化：小寫、去標點、縮寫展開、數字統一、英美拼法統一、移除贅詞（uh, um…）。

    參數:
        text: 原始文字（ref 或 hyp）。

    回傳:
        正規化後文字，詞與詞之間以單一空白分隔；空字串輸入回傳空字串。
    """
    raise NotImplementedError
