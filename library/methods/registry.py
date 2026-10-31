"""方法註冊與 get_method()（支援 "vad+denoise" 組合）。"""

from library.methods.base import AudioMethod
from library.methods.chain import MethodChain
from library.methods.decoding import BeamMethod, GreedyMethod
from library.methods.denoise import DenoiseMethod
from library.methods.identity import IdentityMethod
from library.methods.loudnorm import LoudnormMethod
from library.methods.vad import VADMethod
from library.utils.paths import configs_dir
from library.utils.yaml_io import load_yaml

_REGISTRY: dict[str, type[AudioMethod]] = {}


def _default_config(name: str) -> dict:
    path = configs_dir() / "methods" / f"{name}.yaml"
    return load_yaml(path) if path.is_file() else {}


def get_method(spec: str, config: dict | None = None) -> AudioMethod:
    """spec 可為單一名稱或以 '+' 串接，例如 'vad+denoise' → MethodChain([VAD, Denoise])。
    串接順序即處理順序。

    參數:
        spec: 方法名稱或組合（見 plan.md 6.5）。
        config: {方法名稱: 設定 dict}；未提供的方法讀取 configs/methods/<name>.yaml（不存在則用預設值）。

    回傳:
        AudioMethod 實例（組合時為 MethodChain）。

    例外:
        KeyError: 有方法名稱未註冊。
        ValueError: spec 為空或含空的組合項目。
    """
    names = spec.split("+")
    if not spec or any(not n for n in names):
        raise ValueError(f"方法組合格式錯誤：{spec!r}")
    unknown = [n for n in names if n not in _REGISTRY]
    if unknown:
        raise KeyError(f"未註冊的方法：{unknown}（可用：{list_methods()}）")
    config = config or {}
    methods = [_REGISTRY[n](config.get(n, _default_config(n))) for n in names]
    return methods[0] if len(methods) == 1 else MethodChain(methods)


def list_methods() -> list[str]:
    """回傳所有已註冊的單一方法名稱。"""
    return list(_REGISTRY)


def register_method(name: str, cls: type[AudioMethod]) -> None:
    """註冊方法。

    參數:
        name: 方法名稱（不可含 '+'）。
        cls: AudioMethod 子類別。

    例外:
        ValueError: 名稱已註冊或含 '+'。
    """
    if "+" in name:
        raise ValueError(f"方法名稱不可含 '+'：{name}")
    if name in _REGISTRY:
        raise ValueError(f"方法名稱已註冊：{name}")
    _REGISTRY[name] = cls


for _cls in (IdentityMethod, VADMethod, DenoiseMethod, LoudnormMethod, BeamMethod, GreedyMethod):
    register_method(_cls.name, _cls)
