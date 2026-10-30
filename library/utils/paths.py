"""專案路徑集中管理。

所有路徑一律使用 pathlib.Path；設定檔中的相對路徑一律相對於專案根目錄解析。
專案路徑含空格（D:\\stt test），不可手動拼接字串。
"""

from pathlib import Path

_ROOT_MARKER = "pyproject.toml"


def project_root() -> Path:
    """回傳專案根目錄（含 pyproject.toml 的資料夾）。

    回傳:
        專案根目錄的絕對路徑。

    例外:
        RuntimeError: 找不到專案根目錄。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _ROOT_MARKER).is_file():
            return parent
    raise RuntimeError(f"找不到專案根目錄（往上找不到 {_ROOT_MARKER}）：{here}")


def resolve_path(path: str | Path) -> Path:
    """將設定檔中的路徑解析為絕對路徑。

    參數:
        path: 絕對路徑，或相對於專案根目錄的路徑。

    回傳:
        絕對路徑（不檢查是否存在）。
    """
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def configs_dir() -> Path:
    """回傳 configs/ 資料夾路徑。"""
    return project_root() / "configs"


def data_dir() -> Path:
    """回傳 data/ 資料夾路徑。"""
    return project_root() / "data"


def results_dir() -> Path:
    """回傳 results/ 資料夾路徑。"""
    return project_root() / "results"
