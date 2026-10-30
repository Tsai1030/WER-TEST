"""run_experiment() 主流程。"""

import logging
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from library.data import Utterance, read_audio, write_audio
from library.export import (
    ResourceRow,
    ResultRow,
    ResultWriter,
    result_key,
    write_meta,
    write_resources,
)
from library.methods import AudioMethod, get_method
from library.models import ASRModel, get_model
from library.monitor import ResourceMonitor, Timer
from library.runner.checkpoint import Checkpoint
from library.runner.config import ExperimentConfig
from library.runner.matrix import degraded_path, iter_matrix, split_utterances
from library.scoring import align
from library.textnorm import EXTRA_FILLERS, get_normalizer
from library.utils.yaml_io import load_yaml

log = logging.getLogger(__name__)

OUTPUT_FILES = ("results.csv", "resources.csv", "meta.json", "checkpoint.json")
_SAVE_EVERY = 20  # 每寫幾列存一次 checkpoint（results.csv 本身逐列 flush）


def _done_in_results(path: Path) -> set[str]:
    """由既有 results.csv 取出已完成的組合鍵（results.csv 是逐列結果的依據）。"""
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False,
                     usecols=["model", "method", "condition", "utt_id"])
    return {result_key(*r) for r in df[["model", "method", "condition", "utt_id"]].itertuples(index=False)}


def _backup_outputs(out_dir: Path) -> None:
    """--no-resume：既有輸出改名為 .bak（覆寫上一份 .bak），不直接刪除。"""
    for name in OUTPUT_FILES:
        p = out_dir / name
        if p.is_file():
            os.replace(p, p.with_name(p.name + ".bak"))
            log.warning("已備份舊輸出：%s → %s.bak", p.name, p.name)


def _run_one(
    cfg: ExperimentConfig,
    model: ASRModel,
    method: AudioMethod,
    method_spec: str,
    cond: str,
    u: Utterance,
    normalizer: Callable[[str], str],
) -> ResultRow:
    """跑一個「模型 × 方法 × 條件 × 句子」：讀劣化音檔 → 方法 → 辨識 → 正規化 → 對齊。"""
    wav, sr = read_audio(degraded_path(cfg, cond, u.utt_id))
    with Timer() as t_method:
        processed = method.process(wav, sr)
    if len(processed):
        with Timer() as t_asr:
            tr = model.transcribe(processed, sr)
        hyp, lang, asr_ms = tr.text, tr.language or "", t_asr.ms
    else:  # VAD 判斷無語音：不呼叫模型，記錄空字串（plan.md 5.4）
        hyp, lang, asr_ms = "", "", 0.0
    if cfg.save_processed and len(processed):
        dst = cfg.output_dir / "processed" / method_spec / cond / f"{u.utt_id}.wav"
        if not dst.is_file():  # 方法處理與模型無關，存一份即可
            write_audio(dst, processed, sr)
    ref_norm, hyp_norm = normalizer(u.ref_raw), normalizer(hyp)
    r = align(ref_norm, hyp_norm)
    return ResultRow(
        model=model.name, condition=cond, method=method_spec, utt_id=u.utt_id,
        speaker_id=u.speaker_id, gender=u.gender, audio_sec=u.audio_sec,
        ref_raw=u.ref_raw, hyp_raw=hyp, ref_norm=ref_norm, hyp_norm=hyp_norm,
        wer=r.wer, S=r.S, D=r.D, I=r.I, N=r.N,
        latency_ms=round(t_method.ms + asr_ms, 2), method_ms=round(t_method.ms, 2), detected_lang=lang,
    )


def _meta_extra(cfg: ExperimentConfig, speech: list[Utterance], noise: list[Utterance],
                ckpt: Checkpoint, resumed: bool) -> dict:
    cond_cfg = load_yaml(cfg.conditions_file)
    n_female = sum(u.gender == "female" for u in speech)
    conditions = {}
    for c in cfg.conditions:
        spec = dict(cond_cfg["conditions"][c] or {})
        if spec.get("type") in ("noise", "noise_only"):
            spec["noise_source"] = f"{cond_cfg['noise_source']} {cond_cfg.get('noise_files', [])}"
        conditions[c] = spec
    return {
        "dataset": f"{cfg.dataset}, {n_female}+{len(speech) - n_female}, seed {cfg.seed}, + {len(noise)} noise-only",
        "normalizer": f"{cfg.normalizer}: whisper EnglishTextNormalizer + filler removal {list(EXTRA_FILLERS)}",
        "models": ckpt.models,
        "methods": {spec: get_method(spec).describe() for spec in cfg.methods},
        "conditions": conditions,
        "matrix": {"models": cfg.models, "methods": cfg.methods, "conditions": cfg.conditions,
                   "n_speech": len(speech), "n_noise": len(noise)},
        "resumed": resumed,
    }


def run_experiment(cfg: ExperimentConfig, resume: bool = True) -> Path:
    """執行完整實驗矩陣，逐列寫入 results.csv，結束後寫 resources.csv 與 meta.json。

    - 劣化音檔從 cfg.degraded_dir 讀取（預先產生，不即時產生）。
    - 方法回傳空音訊時不呼叫模型，直接記錄空字串 hyp。
    - 每換一個方法重設峰值記憶體統計；每個模型結束後 unload。
    - cfg.save_processed 為 true 時才把方法處理後的音訊存檔。
    - 續跑：已完成的組合 = checkpoint.json ∪ results.csv 中已有的列，不會重複寫入。
      完全跑完的模型不會再載入；其資源量測與描述取自 checkpoint。

    參數:
        cfg: 實驗設定。
        resume: 是否依 checkpoint.json 跳過已完成組合；False 時舊輸出改名為 .bak 後重跑。

    回傳:
        結果資料夾路徑（cfg.output_dir）。

    例外:
        FileNotFoundError: manifest 或劣化音檔不存在。
    """
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        _backup_outputs(out)
    ckpt = Checkpoint(out / "checkpoint.json")
    ckpt.done |= _done_in_results(out / "results.csv")
    resumed = bool(ckpt.done)
    normalizer = get_normalizer(cfg.normalizer)
    speech, noise = split_utterances(cfg)

    todo: dict[tuple[str, str], list[tuple[str, Utterance]]] = {}
    for model_name, method_spec, cond, u in iter_matrix(cfg):
        if result_key(model_name, method_spec, cond, u.utt_id) not in ckpt.done:
            todo.setdefault((model_name, method_spec), []).append((cond, u))
    total = sum(map(len, todo.values()))
    log.info("%s：待執行 %d 筆（已完成 %d 筆）→ %s", cfg.name, total, len(ckpt.done), out)

    written = 0
    try:
        with ResultWriter(out / "results.csv") as writer, tqdm(total=total, unit="筆", dynamic_ncols=True) as bar:
            for model_name in cfg.models:
                pending = [m for m in cfg.methods if (model_name, m) in todo]
                if not pending:
                    continue
                model = get_model(model_name)
                with Timer() as t_load:
                    model.load(cfg.device, cfg.dtype, cfg.num_threads)
                ckpt.models[model_name] = {**model.describe(), "load_time_s": round(t_load.ms / 1000, 2)}
                size_mb = model.weights_size_mb()
                try:
                    for method_spec in pending:
                        bar.set_description(f"{model_name} {method_spec}")
                        method = get_method(method_spec)
                        method.load(cfg.device)
                        with ResourceMonitor() as mon:
                            for cond, u in todo[(model_name, method_spec)]:
                                writer.write(_run_one(cfg, model, method, method_spec, cond, u, normalizer))
                                ckpt.mark_done(model_name, method_spec, cond, u.utt_id)
                                written += 1
                                bar.update(1)
                                if written % _SAVE_EVERY == 0:
                                    ckpt.save()
                        method.unload()
                        ckpt.resources[f"{model_name}|{method_spec}"] = asdict(ResourceRow(
                            model=model_name, method=method_spec, peak_gpu_mb=mon.peak_gpu_mb,
                            peak_ram_mb=round(mon.peak_ram_mb, 1), model_size_mb=size_mb,
                            load_time_s=round(t_load.ms / 1000, 2), gpu_measure=mon.gpu_measure,
                        ))
                        ckpt.save()
                        log.info("完成 %s × %s：峰值 RAM %.0f MB", model_name, method_spec, mon.peak_ram_mb)
                finally:
                    model.unload()
    finally:
        ckpt.save()

    resources = [ResourceRow(**ckpt.resources[k]) for m in cfg.models for s in cfg.methods
                 if (k := f"{m}|{s}") in ckpt.resources]
    write_resources(resources, out / "resources.csv")
    write_meta(cfg, [], out / "meta.json", extra=_meta_extra(cfg, speech, noise, ckpt, resumed))
    log.info("輸出完成：%s（本次寫入 %d 筆）", out, written)
    return out
