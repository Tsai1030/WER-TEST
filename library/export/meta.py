"""收集環境資訊並寫 meta.json（utf-8）。"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from library.utils.paths import project_root

# 只用於型別標註，執行期不 import，維持 export 只依賴 utils 的分層原則
if TYPE_CHECKING:
    from library.models import ASRModel
    from library.runner import ExperimentConfig

TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchaudio",
    "transformers",
    "numpy",
    "librosa",
    "soundfile",
    "jiwer",
    "whisper-normalizer",
    "faster-whisper",
    "silero-vad",
    "noisereduce",
    "pyloudnorm",
    "pyroomacoustics",
)


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _os_name() -> str:
    """作業系統名稱；Windows 11 的 platform.release() 仍回傳 "10"，以 build ≥ 22000 判斷。"""
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        return f"Windows {'11' if build >= 22000 else platform.release()} (build {build})"
    return f"{platform.system()} {platform.release()}"


def _gpu_name() -> str:
    """有 CUDA 時回傳 GPU 名稱，否則 "none"。只在 torch 已安裝時檢查。"""
    try:
        import torch
    except ImportError:
        return "none"
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"


def git_commit(root: Path | None = None) -> str | None:
    """讀取目前的 git commit（短雜湊）；不是 git repo 時回傳 None。直接讀 .git，不呼叫 shell。"""
    git_dir = (root or project_root()) / ".git"
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text(encoding="utf-8").strip()
    if not ref.startswith("ref: "):
        return ref[:7]
    ref_path = git_dir / ref[5:]
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()[:7]
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref[5:]):
                return line.split()[0][:7]
    return None


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """回傳套件版本；未安裝為 None。"""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_environment() -> dict:
    """收集執行環境（meta.json 的環境欄位）：CPU、核心數、RAM、GPU、OS、Python、套件版本、git commit。"""
    return {
        "git_commit": git_commit(),
        "cpu": _cpu_name(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": _gpu_name(),
        "env": _os_name(),
        "python": platform.python_version(),
        "packages": package_versions(),
    }


def write_meta(
    cfg: ExperimentConfig, models: list[ASRModel], path: Path, extra: dict | None = None
) -> None:
    """收集 CPU / GPU、RAM、Python 與套件版本、git commit、設定，寫出 meta.json（utf-8）。

    欄位見 plan.md 6.4；各模型資訊取自 ASRModel.describe()。
    extra 由 runner 提供本模組無法取得的資訊（dataset、normalizer、methods、conditions、句數、續跑紀錄等），
    同名鍵會覆寫預設值；extra["models"] 會合併進各模型的描述。

    參數:
        cfg: 實驗設定。
        models: 本輪使用的模型（用於 describe()）。
        path: 輸出路徑。
        extra: 額外欄位。
    """
    extra = dict(extra or {})
    env = collect_environment()
    model_info = {m.name: m.describe() for m in models}
    for name, info in (extra.pop("models", None) or {}).items():
        model_info[name] = {**model_info.get(name, {}), **info}
    meta = {
        "round": cfg.name,
        "date": date.today().isoformat(),
        "git_commit": env["git_commit"],
        "device": cfg.device,
        "cpu": f"{env['cpu']} ({env['cpu_cores']}C/{env['cpu_threads']}T)",
        "ram_gb": env["ram_gb"],
        "gpu": env["gpu"],
        "num_threads": cfg.num_threads,
        "env": env["env"],
        "python": env["python"],
        "precision": cfg.dtype,
        "dataset": "",
        "normalizer": cfg.normalizer,
        "nano_budget_mb": cfg.nano_budget_mb,
        "seed": cfg.seed,
        "packages": {k: v for k, v in env["packages"].items() if v is not None},
        "models": model_info,
        "methods": {},
        "conditions": {},
        **extra,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
