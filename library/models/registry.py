"""模型註冊與 get_model()。"""

from library.models.base import ASRModel


def get_model(name: str, config: dict | None = None) -> ASRModel:
    """依名稱建立模型實例（尚未載入權重）。

    參數:
        name: 模型名稱（見 plan.md 6.5）。
        config: configs/models/<name>.yaml 讀入後的 dict；None 表示讀取預設設定檔。

    回傳:
        ASRModel 實例。

    例外:
        KeyError: 名稱未註冊。
        RuntimeError: 設定檔 enabled 為 false（例：parakeet-tdt-v2）。
    """
    raise NotImplementedError


def list_models() -> list[str]:
    """回傳所有已註冊的模型名稱。"""
    raise NotImplementedError


def register_model(name: str, cls: type[ASRModel]) -> None:
    """註冊模型。

    參數:
        name: 模型名稱。
        cls: ASRModel 子類別。

    例外:
        ValueError: 名稱已註冊。
    """
    raise NotImplementedError
