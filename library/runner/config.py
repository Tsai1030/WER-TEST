"""讀取並驗證 YAML 設定。"""

from dataclasses import dataclass
from pathlib import Path


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
    device: str = "cpu"  # Jetson 上改為 "cuda"
    dtype: str = "fp32"  # Jetson 部署預計為 q4，另行複驗（plan.md 11.5）
    num_threads: int = 4  # CPU 執行緒數（i3-10110U 為 2 核 4 緒）
    seed: int = 42
    save_processed: bool = False


def load_config(path: str | Path) -> ExperimentConfig:
    """讀取 YAML（utf-8）並驗證欄位；路徑相對於專案根目錄解析。

    驗證項目：必要欄位存在、models / methods / conditions 名稱符合 plan.md 6.5、
    模型設定檔 enabled 為 true。

    參數:
        path: 設定檔路徑（例：configs/round1.yaml）。

    回傳:
        ExperimentConfig。

    例外:
        FileNotFoundError: 設定檔不存在。
        ValueError: 欄位缺漏或名稱不合法。
    """
    raise NotImplementedError
