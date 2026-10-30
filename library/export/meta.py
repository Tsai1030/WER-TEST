"""收集環境資訊並寫 meta.json（utf-8）。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

# 只用於型別標註，執行期不 import，維持 export 只依賴 utils 的分層原則
if TYPE_CHECKING:
    from library.models import ASRModel
    from library.runner import ExperimentConfig


def write_meta(cfg: ExperimentConfig, models: list[ASRModel], path: Path) -> None:
    """收集 CPU / GPU、RAM、Python 與套件版本、git commit、設定，寫出 meta.json。

    欄位見 plan.md 6.4；各模型資訊取自 ASRModel.describe()。

    參數:
        cfg: 實驗設定。
        models: 本輪使用的模型（用於 describe()）。
        path: 輸出路徑。
    """
    raise NotImplementedError
