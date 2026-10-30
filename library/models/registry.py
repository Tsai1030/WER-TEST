"""模型註冊與 get_model()。"""

from library.models.base import ASRModel
from library.models.granite import GraniteModel
from library.models.parakeet import ParakeetModel
from library.models.qwen3_asr import Qwen3ASRModel
from library.models.whisper import WhisperModel
from library.utils.paths import configs_dir
from library.utils.yaml_io import load_yaml

_REGISTRY: dict[str, type[ASRModel]] = {}


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
    if name not in _REGISTRY:
        raise KeyError(f"未註冊的模型：{name}（可用：{list_models()}）")
    if config is None:
        config = load_yaml(configs_dir() / "models" / f"{name}.yaml")
    if not config.get("enabled", True):
        raise RuntimeError(f"模型 {name} 在設定檔中為 enabled: false")
    return _REGISTRY[name](config)


def list_models() -> list[str]:
    """回傳所有已註冊的模型名稱。"""
    return list(_REGISTRY)


def register_model(name: str, cls: type[ASRModel]) -> None:
    """註冊模型。

    參數:
        name: 模型名稱。
        cls: ASRModel 子類別。

    例外:
        ValueError: 名稱已註冊。
    """
    if name in _REGISTRY:
        raise ValueError(f"模型名稱已註冊：{name}")
    _REGISTRY[name] = cls


for _cls in (GraniteModel, ParakeetModel, WhisperModel, Qwen3ASRModel):
    register_model(_cls.name, _cls)
