"""get_normalizer()，預留中文版。"""

from collections.abc import Callable

from library.textnorm.english import normalize_english

_NORMALIZERS: dict[str, Callable[[str], str]] = {"english": normalize_english}
_LANG_TO_NAME = {"en": "english"}


def normalize(text: str, lang: str = "en") -> str:
    """英文：小寫、去標點、縮寫展開、數字統一、英美拼法統一、移除贅詞（uh, um…）。

    參數:
        text: 原始文字。
        lang: 語言代碼；本輪只支援 "en"。

    回傳:
        正規化後文字。

    例外:
        KeyError: 不支援的語言。
    """
    if lang not in _LANG_TO_NAME:
        raise KeyError(f"不支援的語言：{lang}（可用：{list(_LANG_TO_NAME)}）")
    return get_normalizer(_LANG_TO_NAME[lang])(text)


def get_normalizer(name: str = "english") -> Callable[[str], str]:
    """依名稱取得正規化函式（對應 round1.yaml 的 normalizer 欄）。

    參數:
        name: 正規化器名稱；本輪只有 "english"，預留中文版。

    回傳:
        接收原始文字、回傳正規化文字的函式。

    例外:
        KeyError: 名稱未註冊。
    """
    if name not in _NORMALIZERS:
        raise KeyError(f"未註冊的正規化器：{name}（可用：{list(_NORMALIZERS)}）")
    return _NORMALIZERS[name]
