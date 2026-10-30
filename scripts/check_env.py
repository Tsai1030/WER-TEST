"""檢查 Python、套件版本、可用裝置（CPU / CUDA）、模型能否載入，並實測每句秒數。

用法:
    uv run python scripts/check_env.py --bench 5
    uv run python scripts/check_env.py --models granite-470m --bench 5
"""

import argparse
import importlib
import os
import shutil

from library.data import load_manifest
from library.export import collect_environment
from library.runner import bench_model, inferences_per_model
from library.utils import load_yaml, resolve_path, setup_logging

# 選用套件：import 名稱 →（用途, 是否為目前設定所必需）
OPTIONAL_IMPORTS = {
    "silero_vad": ("vad（silero-vad）", True),
    "noisereduce": ("denoise（noisereduce，目前 impl）", True),
    "pyloudnorm": ("loudnorm（pyloudnorm）", True),
    "pyroomacoustics": ("reverb（pyroomacoustics）", True),
    "faster_whisper": ("whisper int8 備案（faster-whisper）", True),
    "df.enhance": ("denoise 原規劃（deepfilternet，已停用，僅供參考）", False),
}

# plan.md 8.5：預估總時間 → 做法
SCALE_RULES = (
    (10.0, "≤ 10 小時：照原矩陣，夜間執行"),
    (24.0, "10–24 小時：條件縮為 clean、snr10、snr00、reverb、lowvol、noise_only；Whisper 改 faster-whisper int8"),
    (float("inf"), "> 24 小時：再減為每性別 15 句（共 30 句），或分模型分天執行"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round1.yaml", help="實驗設定檔")
    parser.add_argument("--models", nargs="*", default=None, help="要檢查的模型；預設為設定檔中的全部")
    parser.add_argument("--bench", type=int, default=0, help="每個模型實測幾句 clean 音檔（0 = 不實測）")
    parser.add_argument(
        "--whisper-framework",
        choices=["transformers", "faster-whisper"],
        default=None,
        help="暫時覆寫 whisper-small 的 framework（不修改設定檔），用於比較速度",
    )
    return parser.parse_args(argv)


def _print_environment() -> None:
    env = collect_environment()
    print("== 環境")
    for key in ("cpu", "cpu_cores", "cpu_threads", "ram_gb", "gpu", "env", "python", "git_commit"):
        print(f"  {key:12s} {env[key]}")
    print(f"  {'HF_HOME':12s} {os.environ.get('HF_HOME') or '（未設定，使用預設快取）'}")
    print(f"  {'ffmpeg':12s} {shutil.which('ffmpeg') or '（未安裝）'}")
    import torch

    print(f"  {'torch':12s} cuda_available={torch.cuda.is_available()} threads={torch.get_num_threads()}")
    print("== 套件版本")
    for name, version in env["packages"].items():
        print(f"  {name:16s} {version or '（未安裝）'}")


def _check_optional_imports() -> list[str]:
    print("== 選用套件 import")
    failed = []
    for module, (purpose, required) in OPTIONAL_IMPORTS.items():
        try:
            importlib.import_module(module)
            print(f"  OK    {module:16s} {purpose}")
        except Exception as exc:  # noqa: BLE001  套件內部錯誤也要回報，不中斷檢查
            if required:
                failed.append(module)
            status = "FAIL" if required else "WARN"
            print(f"  {status}  {module:16s} {purpose}：{type(exc).__name__}: {exc}")
    return failed


def main(argv: list[str] | None = None) -> int:
    """執行環境檢查；回傳 0 代表全部通過。"""
    args = parse_args(argv)
    setup_logging("WARNING")
    cfg = load_yaml(resolve_path(args.config))
    models = args.models if args.models is not None else cfg["models"]

    _print_environment()
    failed_imports = _check_optional_imports()
    if args.bench <= 0:
        return 1 if failed_imports else 0

    utts = load_manifest(resolve_path(cfg["paths"]["manifest"]))[: args.bench + 1]
    cond_cfg = load_yaml(resolve_path("configs/conditions.yaml"))
    n_noise = cond_cfg["conditions"].get("noise_only", {}).get("n_clips", 0)
    n_speech = cfg["sampling"]["n_per_gender"] * 2
    per_model = inferences_per_model(len(cfg["methods"]), cfg["conditions"], n_speech, n_noise)

    print(f"== 模型實測（暖機 1 句 + 計時 {len(utts) - 1} 句，device={cfg['device']}, dtype={cfg['dtype']}, "
          f"threads={cfg['num_threads']}）")
    results, failed_models = [], []
    for name in models:
        model_cfg = load_yaml(resolve_path(f"configs/models/{name}.yaml"))
        is_whisper = name.startswith("whisper-")
        if is_whisper and args.whisper_framework:
            model_cfg["framework"] = args.whisper_framework
        label = f"{name}（{model_cfg['framework']}）" if is_whisper else name
        print(f"-- {label}")
        try:
            r = bench_model(name, utts, cfg["device"], cfg["dtype"], cfg["num_threads"], model_cfg)
        except Exception as exc:  # noqa: BLE001  單一模型失敗不影響其他模型
            failed_models.append(name)
            print(f"  FAIL  {type(exc).__name__}: {exc}")
            continue
        results.append(r)
        print(f"  載入 {r.load_s:.1f}s、載入後 RSS {r.rss_after_load_mb:.0f} MB、暖機 {r.warmup_ms / 1000:.1f}s")
        for (utt_id, ref, hyp), ms, sec in zip(r.samples, r.utt_ms, r.audio_sec, strict=True):
            print(f"  {utt_id} {ms / 1000:5.1f}s / 音長 {sec:.1f}s")
            print(f"    ref: {ref}")
            print(f"    hyp: {hyp}")

    if results:
        print(f"== 估算（每模型 {per_model} 次推論；未含方法處理時間）")
        total_h = 0.0
        for r in results:
            hours = r.mean_sec * per_model / 3600
            total_h += hours
            print(f"  {r.model:16s} 平均 {r.mean_sec:5.2f} s/句  RTF {r.rtf:.2f}  → {hours:5.1f} 小時")
        print(f"  合計 {total_h:.1f} 小時")
        rule = next(text for limit, text in SCALE_RULES if total_h <= limit)
        print(f"  依 8.5 節：{rule}")
    return 1 if failed_imports or failed_models else 0


if __name__ == "__main__":
    raise SystemExit(main())
