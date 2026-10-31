"""讀取並驗證 YAML 設定。"""

from dataclasses import dataclass, field
from pathlib import Path

from library.export.schema import CONDITION_NAMES, METHOD_NAMES, MODEL_NAMES
from library.methods import get_method
from library.models import get_model
from library.utils.paths import configs_dir, resolve_path
from library.utils.yaml_io import load_yaml


@dataclass
class ExperimentConfig:
    """一輪實驗的設定（來自 configs/round1.yaml 等）。"""

    name: str
    manifest: Path
    degraded_dir: Path
    output_dir: Path
    models: list[str]
    methods: list[str]
    conditions: list[str]
    device: str = "cpu"  # Jetson 部署預計為 q4，另行複驗（plan.md 11.5）
    dtype: str = "fp32"
    num_threads: int = 4  # CPU 執行緒數（i3-10110U 為 2 核 4 緒）
    seed: int = 42
    save_processed: bool = False
    normalizer: str = "english"
    conditions_file: Path = field(default_factory=lambda: configs_dir() / "conditions.yaml")
    max_utterances: int | None = None  # 只取前 N 句語音（冒煙 / 除錯用；None = 全部）
    dataset: str = ""
    nano_budget_mb: int = 8192
    # {方法名稱: 設定}；取代 configs/methods/<name>.yaml，只作用於本輪（plan.md 7.1）
    method_params: dict[str, dict] = field(default_factory=dict)
    # 不執行的 (模型, 方法)：解碼方法與 baseline 相同或模型不支援（plan.md 7.1）
    exclude: list[tuple[str, str]] = field(default_factory=list)


def _check_names(kind: str, names: list[str], allowed: tuple[str, ...], split_plus: bool = False) -> None:
    if not names:
        raise ValueError(f"設定檔的 {kind} 不可為空")
    if len(set(names)) != len(names):
        raise ValueError(f"設定檔的 {kind} 有重複：{names}")
    for n in names:
        parts = n.split("+") if split_plus else [n]
        bad = [p for p in parts if p not in allowed]
        if bad:
            raise ValueError(f"{kind} 名稱 {n!r} 不符 plan.md 6.5 命名規則（可用：{allowed}）")


def _parse_exclude(raw: list | None, cfg: ExperimentConfig) -> list[tuple[str, str]]:
    pairs = []
    for item in raw or []:
        if not isinstance(item, dict) or set(item) != {"model", "method"}:
            raise ValueError(f"exclude 每項需為 {{model, method}}：{item!r}")
        pair = (item["model"], item["method"])
        if pair[0] not in cfg.models or pair[1] not in cfg.methods:
            raise ValueError(f"exclude {pair} 不在本輪的 models / methods 中")
        if pair[1] == "baseline":
            raise ValueError("exclude 不可排除 baseline（ΔWER 需要 baseline）")
        if pair in pairs:
            raise ValueError(f"exclude 有重複：{pair}")
        pairs.append(pair)
    return pairs


def _check_decoding(cfg: ExperimentConfig) -> None:
    for m in cfg.models:
        supported = set(get_model(m).supported_decoding)
        for spec in cfg.methods:
            if (m, spec) in cfg.exclude:
                continue
            bad = sorted(set(get_method(spec, cfg.method_params).decoding()) - supported)
            if bad:
                raise ValueError(f"{m} 不支援 {spec} 的解碼參數 {bad}；請在 exclude 排除此組合")


def load_config(path: str | Path) -> ExperimentConfig:
    """讀取 YAML（utf-8）並驗證欄位；路徑相對於專案根目錄解析。

    驗證項目：必要欄位存在、models / methods / conditions / method_params 名稱符合 plan.md 6.5、
    模型設定檔 enabled 為 true、每個條件都定義在 conditions.yaml、exclude 格式正確、
    未排除的組合中模型支援方法的解碼參數。

    參數:
        path: 設定檔路徑（例：configs/round1.yaml）。

    回傳:
        ExperimentConfig。

    例外:
        FileNotFoundError: 設定檔不存在。
        ValueError: 欄位缺漏或名稱不合法。
    """
    raw = load_yaml(resolve_path(path))
    try:
        paths = raw["paths"]
        cfg = ExperimentConfig(
            name=raw["name"],
            manifest=resolve_path(paths["manifest"]),
            degraded_dir=resolve_path(paths["degraded_dir"]),
            output_dir=resolve_path(paths["output_dir"]),
            models=list(raw["models"]),
            methods=list(raw["methods"]),
            conditions=list(raw["conditions"]),
        )
    except KeyError as exc:
        raise ValueError(f"設定檔缺少欄位：{exc}") from exc
    for key in ("device", "dtype", "num_threads", "seed", "save_processed", "normalizer", "max_utterances"):
        if key in raw:
            setattr(cfg, key, raw[key])
    if "conditions_file" in raw:
        cfg.conditions_file = resolve_path(raw["conditions_file"])
    cfg.method_params = dict(raw.get("method_params") or {})
    meta = raw.get("meta") or {}
    cfg.dataset = meta.get("dataset", cfg.dataset)
    cfg.nano_budget_mb = meta.get("nano_budget_mb", cfg.nano_budget_mb)

    _check_names("models", cfg.models, MODEL_NAMES)
    _check_names("methods", cfg.methods, METHOD_NAMES, split_plus=True)
    _check_names("conditions", cfg.conditions, CONDITION_NAMES)
    if cfg.method_params:
        _check_names("method_params", list(cfg.method_params), METHOD_NAMES)
    for m in cfg.models:
        model_cfg = load_yaml(configs_dir() / "models" / f"{m}.yaml")
        if not model_cfg.get("enabled", True):
            raise ValueError(f"模型 {m} 在設定檔中為 enabled: false，不可列入實驗")
    defined = load_yaml(cfg.conditions_file).get("conditions", {})
    undefined = [c for c in cfg.conditions if c not in defined]
    if undefined:
        raise ValueError(f"條件未定義在 {cfg.conditions_file.name}：{undefined}")
    cfg.exclude = _parse_exclude(raw.get("exclude"), cfg)
    _check_decoding(cfg)
    return cfg
