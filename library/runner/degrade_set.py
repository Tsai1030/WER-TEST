"""M2：依 manifest 產生所有劣化條件音檔，並逐檔驗證（實測 SNR、增益、RT60）。"""

from pathlib import Path

import numpy as np
import pandas as pd

from library.data import load_manifest, read_audio, write_audio, write_noise_entries
from library.degrade import (
    apply_condition,
    build_noise_only_clips,
    estimate_rt60,
    pick_noise_env,
    room_impulse_response,
    speech_component,
)

LOG_NAME = "degrade_log.csv"


def _rms_db(wav: np.ndarray) -> float:
    return 10 * np.log10(float(np.mean(wav.astype(np.float64) ** 2)))


def _snr_db(speech_part: np.ndarray, written: np.ndarray) -> float:
    """已知語音成分時的實際 SNR（噪音成分 = 寫出的音檔 − 語音成分，含 16-bit 量化）。"""
    noise_part = written.astype(np.float64) - speech_part
    return 10 * np.log10(float(np.dot(speech_part, speech_part)) / float(np.dot(noise_part, noise_part)))


def make_degraded_set(
    manifest_path: Path,
    out_root: Path,
    conditions: list[str],
    cond_cfg: dict,
    seed: int,
) -> pd.DataFrame:
    """產生 out_root/<condition>/<utt_id>.wav 與純噪音片段，回傳逐檔驗證紀錄。

    - 每句的 seed 為 seed + 句子序號；同一句在各 SNR 條件使用相同噪音環境與片段。
    - 寫檔後重新讀取（含 16-bit 量化）再量測：noise 條件量 SNR（以已知語音成分計算，不估計增益）、
      volume 量增益、reverb 量 RIR 的 RT60。
    - noise_only 片段寫到 out_root/noise_only/noiseNN.wav，並更新 manifest 的 noise 列。
    - 驗證紀錄另存為 out_root/degrade_log.csv（utf-8）；已存在時只取代本次條件的列，其他條件保留。

    參數:
        manifest_path: data/clean/manifest.csv。
        out_root: 劣化音檔根目錄（data/degraded）。
        conditions: 條件名稱（round1.yaml 的 conditions，可含 noise_only）。
        cond_cfg: configs/conditions.yaml 讀入後的 dict。
        seed: 亂數種子。

    回傳:
        本次產生的條件的驗證紀錄（DataFrame），欄位 utt_id、condition、type、noise_env、target、measured、error。

    例外:
        FileNotFoundError: manifest 或噪音源不存在。
        KeyError: 條件不在 conditions.yaml。
    """
    utts = [u for u in load_manifest(manifest_path) if u.utt_id.startswith("utt")]
    speech_conditions = [c for c in conditions if c != "noise_only"]
    rows = []
    for i, u in enumerate(utts):
        speech, sr = read_audio(u.wav_path)
        utt_seed = seed + i
        for cond in speech_conditions:
            spec = cond_cfg["conditions"][cond] or {}
            kind = spec.get("type", "clean")
            dst = out_root / cond / f"{u.utt_id}.wav"
            write_audio(dst, apply_condition(speech, sr, cond, cond_cfg, utt_seed), sr)
            written, _ = read_audio(dst)
            row = {"utt_id": u.utt_id, "condition": cond, "type": kind, "noise_env": "",
                   "target": np.nan, "measured": np.nan}
            if kind == "noise":
                row.update(noise_env=pick_noise_env(cond_cfg, utt_seed), target=spec["snr_db"],
                           measured=_snr_db(speech_component(speech, sr, cond, cond_cfg, utt_seed), written))
            elif kind == "volume":
                row.update(target=spec["gain_db"], measured=_rms_db(written) - _rms_db(speech))
            elif kind == "reverb":
                rir = room_impulse_response(sr, spec["rt60"], tuple(spec["room_dim"]),
                                            spec["src_mic_distance"],
                                            int(spec.get("geometry_seed", utt_seed)))
                row.update(target=spec["rt60"], measured=estimate_rt60(rir, sr))
            else:
                row.update(target=0.0, measured=float(np.max(np.abs(written - speech))))
            rows.append(row)

    if "noise_only" in conditions:
        entries = []
        for k, (env, clip, sr) in enumerate(build_noise_only_clips(cond_cfg, seed), start=1):
            utt_id = f"noise{k:02d}"
            dst = out_root / "noise_only" / f"{utt_id}.wav"
            write_audio(dst, clip, sr)
            written, _ = read_audio(dst)
            entries.append({"utt_id": utt_id, "wav_path": dst, "audio_sec": len(written) / sr,
                            "source_file": env})
            rows.append({"utt_id": utt_id, "condition": "noise_only", "type": "noise_only",
                         "noise_env": env, "target": cond_cfg["conditions"]["noise_only"]["rms_dbfs"],
                         "measured": _rms_db(written)})
        write_noise_entries(manifest_path, entries)

    log = pd.DataFrame(rows)
    log["error"] = log["measured"] - log["target"]
    # 合併舊紀錄：只取代本次重新產生的條件，其他條件（例如前一輪的）保留
    log_path = out_root / LOG_NAME
    merged = log
    if log_path.is_file():
        old = pd.read_csv(log_path, encoding="utf-8", keep_default_na=False, na_values=[""])
        merged = pd.concat([old[~old["condition"].isin(conditions)], log], ignore_index=True)
    merged.to_csv(log_path, index=False, encoding="utf-8", lineterminator="\n")
    return log
