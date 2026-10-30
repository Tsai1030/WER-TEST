"""library.textnorm：標點、縮寫、數字、贅詞。"""

import pytest

from library.textnorm import EXTRA_FILLERS, get_normalizer, normalize, normalize_english


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("I don't know, uh, twenty-five.", "i do not know 25"),  # library/textnorm/README 範例
        ("Hello, World!", "hello world"),  # 大小寫、標點
        ('He was praised for "singing" the anthem.', "he was praised for singing the anthem"),
        ("It's OK.", "it is ok"),  # 縮寫展開
        ("The file size was only two hundred bytes.", "the file size was only 200 bytes"),  # 數字
        ("The file size was only 200 bytes.", "the file size was only 200 bytes"),
        ("The colour of the theatre is grey.", "the color of the theater is gray"),  # 英美拼法
        ("Um, hmm, uh, I mean.", "i mean"),  # Whisper 內建贅詞
        ("Uhm, erm, umm, uhh, hm, yes.", "yes"),  # 額外贅詞
        ("♪ Thank you for watching ♪", "thank you for watching"),  # 符號
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_english(raw: str, expected: str) -> None:
    assert normalize_english(raw) == expected
    assert normalize(raw) == expected


def test_number_forms_become_equal() -> None:
    """ref 寫成文字、hyp 寫成數字時，正規化後必須相同（避免假錯誤）。"""
    assert normalize("only two hundred bytes") == normalize("only 200 bytes")


def test_fillers_not_removed_inside_words() -> None:
    assert normalize("the hummus was umami") == "the hummus was umami"
    assert normalize("ah er") == "ah er"  # 可能是一般詞彙，不移除
    assert "uhm" in EXTRA_FILLERS


def test_whisper_reads_oh_as_zero() -> None:
    """Whisper normalizer 會把 "oh" 當成數字 0；ref 與 hyp 用同一個 normalizer，不會造成假錯誤。"""
    assert normalize("oh I see") == "0 i see"
    assert normalize("Oh, I see.") == normalize("oh i see")


def test_idempotent() -> None:
    once = normalize("Mr. Smith paid $3.50 for twenty-one colour pencils, um, yesterday.")
    assert normalize(once) == once


def test_single_spaces() -> None:
    assert "  " not in normalize("a ,  b ... c")


def test_registry_errors() -> None:
    assert get_normalizer("english") is normalize_english
    with pytest.raises(KeyError):
        get_normalizer("chinese")
    with pytest.raises(KeyError):
        normalize("你好", lang="zh")
