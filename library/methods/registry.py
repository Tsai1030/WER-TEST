"""方法註冊與 get_method()（支援 "vad+denoise" 組合）。"""

from library.methods.base import AudioMethod


def get_method(spec: str, config: dict | None = None) -> AudioMethod:
    """spec 可為單一名稱或以 '+' 串接，例如 'vad+denoise' → MethodChain([VAD, Denoise])。
    串接順序即處理順序。

    參數:
        spec: 方法名稱或組合（見 plan.md 6.5）。
        config: {方法名稱: 設定 dict}；None 表示使用預設值。

    回傳:
        AudioMethod 實例（組合時為 MethodChain）。

    例外:
        KeyError: 有方法名稱未註冊。
    """
    raise NotImplementedError


def list_methods() -> list[str]:
    """回傳所有已註冊的單一方法名稱。"""
    raise NotImplementedError


def register_method(name: str, cls: type[AudioMethod]) -> None:
    """註冊方法。

    參數:
        name: 方法名稱（不可含 '+'）。
        cls: AudioMethod 子類別。

    例外:
        ValueError: 名稱已註冊或含 '+'。
    """
    raise NotImplementedError
