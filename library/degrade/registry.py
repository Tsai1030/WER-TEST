"""條件名稱 → 處理函式的對應表，以及噪音源管理。

噪音源（conditions.yaml 的 noise_source 資料夾下的 WAV）每個檔案切成兩段：
前段用於 add_noise 疊加，最後 noise_holdout_sec 秒保留給 noise_only，兩者不重疊。
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from library.degrade.noise import add_noise
from library.degrade.noise_only import make_noise_only
from library.degrade.reverb import add_reverb
from library.degrade.volume import change_volume
from library.utils.paths import resolve_path

CONDITION_TYPES: tuple[str, ...] = ("clean", "noise", "reverb", "volume", "noise_only")
CLIP_PEAK = 0.99


@lru_cache(maxsize=4)
def _load_noise_bank(
    source: Path, files: tuple[str, ...] | None, holdout_sec: float
) -> tuple[tuple[str, np.ndarray, np.ndarray, int], ...]:
    """回傳 ((名稱, 疊加用區段, 保留區段, 取樣率), ...)，依名稱排序。"""
    paths = (
        [source / f"{name}.wav" for name in files] if files else sorted(source.glob("*.wav"))
    )
    if not paths:
        raise FileNotFoundError(f"噪音源資料夾沒有 WAV：{source}")
    bank = []
    for p in sorted(paths):
        if not p.is_file():
            raise FileNotFoundError(f"噪音源不存在：{p}")
        wav, sr = sf.read(p, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        cut = len(wav) - round(holdout_sec * sr)
        if cut <= 0:
            raise ValueError(f"噪音源 {p.name} 長度 {len(wav) / sr:.1f}s 不足 holdout {holdout_sec}s")
        bank.append((p.stem, wav[:cut], wav[cut:], sr))
    return tuple(bank)


def noise_bank(cfg: dict) -> tuple[tuple[str, np.ndarray, np.ndarray, int], ...]:
    """依 conditions.yaml 讀入噪音源（快取）。

    回傳:
        ((名稱, 疊加用區段, 保留給 noise_only 的區段, 取樣率), ...)，依名稱排序。

    例外:
        FileNotFoundError: 噪音源資料夾或指定檔案不存在。
        ValueError: 噪音源長度不足 holdout。
    """
    files = cfg.get("noise_files")
    return _load_noise_bank(
        resolve_path(cfg["noise_source"]),
        tuple(files) if files else None,
        float(cfg.get("noise_holdout_sec", 60.0)),
    )


def pick_noise_env(cfg: dict, seed: int) -> str:
    """回傳 seed 對應的噪音環境名稱（與 apply_condition 的選擇一致，用於紀錄）。"""
    bank = noise_bank(cfg)
    return bank[int(np.random.default_rng(seed).integers(len(bank)))][0]


def prevent_clipping(wav: np.ndarray, peak: float = CLIP_PEAK) -> np.ndarray:
    """峰值超過 peak 時整體等比例縮小（不改變 SNR），避免寫成 16-bit 時削波。"""
    m = float(np.max(np.abs(wav))) if len(wav) else 0.0
    return (wav * (peak / m)).astype(np.float32) if m > peak else wav.astype(np.float32)


def _condition_spec(cfg: dict, condition: str) -> dict:
    conditions = cfg.get("conditions", {})
    if condition not in conditions:
        raise KeyError(f"conditions.yaml 沒有條件：{condition}（可用：{list(conditions)}）")
    spec = dict(conditions[condition] or {})
    spec.setdefault("type", "clean")
    if spec["type"] not in CONDITION_TYPES:
        raise KeyError(f"未註冊的條件類型：{spec['type']}（可用：{CONDITION_TYPES}）")
    return spec


def apply_condition(speech: np.ndarray, sr: int, condition: str, cfg: dict, seed: int) -> np.ndarray:
    """依條件名稱（clean / snr20 / reverb …）套用對應處理。

    - clean：原樣回傳。
    - noise：以 seed 選噪音環境與片段（同一句在各 SNR 用同一片段），疊加到 snr_db，再防削波。
    - reverb：RIR 擺放位置由 geometry_seed 決定（未設定時用 seed），再防削波。
    - volume：增益 gain_db。

    參數:
        speech: 語音陣列（一維 float32）。
        sr: 取樣率。
        condition: 條件名稱（見 plan.md 6.5）。
        cfg: configs/conditions.yaml 讀入後的 dict（含 noise_source 與 conditions）。
        seed: 每句的亂數種子。

    回傳:
        劣化後的音訊（float32）。

    例外:
        KeyError: 條件名稱不在 cfg 中，或條件類型未註冊。
        ValueError: noise_only 不可用本函式產生（請用 build_noise_only_clips），或噪音源取樣率不符。
    """
    spec = _condition_spec(cfg, condition)
    kind = spec["type"]
    if kind == "clean":
        return speech.astype(np.float32)
    if kind == "noise":
        rng = np.random.default_rng(seed)
        bank = noise_bank(cfg)
        _, mix_part, _, noise_sr = bank[int(rng.integers(len(bank)))]
        if noise_sr != sr:
            raise ValueError(f"噪音源取樣率 {noise_sr} 與語音 {sr} 不符")
        return prevent_clipping(add_noise(speech, mix_part, spec["snr_db"], int(rng.integers(2**31))))
    if kind == "reverb":
        wet = add_reverb(
            speech,
            sr,
            spec["rt60"],
            spec["room_dim"],
            spec["src_mic_distance"],
            int(spec.get("geometry_seed", seed)),
        )
        return prevent_clipping(wet)
    if kind == "volume":
        return change_volume(speech, spec["gain_db"])
    raise ValueError("noise_only 請用 build_noise_only_clips() 產生")


def build_noise_only_clips(cfg: dict, seed: int) -> list[tuple[str, np.ndarray, int]]:
    """由各噪音源的保留區段切出純噪音片段，並調整到 rms_dbfs。

    n_clips 段依噪音環境輪流分配（例：10 段 / 4 個環境 → 3、3、2、2）。

    參數:
        cfg: configs/conditions.yaml 讀入後的 dict（conditions.noise_only 需有 n_clips、clip_sec、rms_dbfs）。
        seed: 亂數種子。

    回傳:
        [(噪音環境名稱, 片段, 取樣率), ...]，共 n_clips 段。

    例外:
        KeyError: conditions 中沒有 noise_only。
        ValueError: 保留區段長度不足。
    """
    spec = _condition_spec(cfg, "noise_only")
    bank = noise_bank(cfg)
    n_clips, clip_sec = int(spec["n_clips"]), float(spec["clip_sec"])
    target_rms = 10 ** (float(spec["rms_dbfs"]) / 20)
    per_env: list[list[tuple[str, np.ndarray, int]]] = []
    for env_idx, (name, _, holdout, sr) in enumerate(bank):
        k = len(range(env_idx, n_clips, len(bank)))
        clips = make_noise_only(holdout, sr, k, clip_sec, seed + env_idx) if k else []
        per_env.append([(name, prevent_clipping(_set_rms(c, target_rms)), sr) for c in clips])
    # 環境輪流排列，讓 noise01、noise02… 交錯來自不同環境
    return [env[r] for r in range(max(map(len, per_env))) for env in per_env if r < len(env)]


def _set_rms(wav: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    return wav * (target_rms / rms) if rms > 0 else wav


def list_conditions() -> list[str]:
    """回傳所有已註冊的條件類型（clean / noise / reverb / volume / noise_only）。

    實際條件名稱（snr20、lowvol…）定義在 conditions.yaml。
    """
    return list(CONDITION_TYPES)
