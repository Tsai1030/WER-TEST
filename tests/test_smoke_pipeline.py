"""冒煙測試：讀劣化音檔 → 前處理方法 → 模型辨識 → 正規化 → 對齊，跑完整流程。

- 預設：granite-470m（最快）× 全部方法 × clean / noise_only，2 句 + 1 段純噪音。
  開發機上光 import torch + transformers 與載入模型就要 35–50 秒，總時間約 1 分鐘出頭。
- SMOKE_ALL=1：round1.yaml 的全部模型 × 全部方法 × 全部條件，2 句 + 2 段純噪音（M4 驗收用，約 20 分鐘）。

需要 M1、M2 的資料與已快取的模型權重；資料不存在時略過。
"""

import os
import time

import pytest

from library.data import load_manifest, read_audio
from library.methods import get_method
from library.models import get_model
from library.scoring import align, corpus_wer
from library.textnorm import normalize
from library.utils import load_yaml, resolve_path

CFG = load_yaml(resolve_path("configs/round1.yaml"))
MANIFEST = resolve_path(CFG["paths"]["manifest"])
DEGRADED = resolve_path(CFG["paths"]["degraded_dir"])
FULL = os.environ.get("SMOKE_ALL") == "1"

MODELS = CFG["models"] if FULL else ["granite-470m"]
METHODS = list(dict.fromkeys(["baseline", "vad", "denoise", "loudnorm", *CFG["methods"]]))
CONDITIONS = CFG["conditions"] if FULL else ["clean", "noise_only"]
N_NOISE = 2 if FULL else 1

pytestmark = pytest.mark.skipif(
    not (MANIFEST.is_file() and (DEGRADED / "noise_only").is_dir()),
    reason="需要 M1 manifest 與 M2 劣化音檔",
)


def _utterances() -> tuple[list, list]:
    utts = load_manifest(MANIFEST)
    speech = [next(u for u in utts if u.gender == g) for g in ("female", "male")]
    noise = [u for u in utts if u.utt_id.startswith("noise")][:N_NOISE]
    return speech, noise


@pytest.mark.parametrize("model_name", MODELS)
def test_pipeline(model_name: str) -> None:
    speech, noise = _utterances()
    model = get_model(model_name)
    model.load(CFG["device"], CFG["dtype"], CFG["num_threads"])
    rows = []
    start = time.perf_counter()
    try:
        for method_name in METHODS:
            method = get_method(method_name)
            method.load(CFG["device"])
            for cond in CONDITIONS:
                for u in noise if cond == "noise_only" else speech:
                    wav, sr = read_audio(DEGRADED / cond / f"{u.utt_id}.wav")
                    processed = method.process(wav, sr)
                    hyp = model.transcribe(processed, sr).text if len(processed) else ""
                    assert isinstance(hyp, str)
                    r = align(normalize(u.ref_raw), normalize(hyp))
                    rows.append((method_name, cond, u.utt_id, r, hyp, len(processed) == 0))
            method.unload()
    finally:
        model.unload()

    elapsed = time.perf_counter() - start
    print(f"\n== {model_name}：{len(rows)} 筆，{elapsed:.0f} 秒")
    for method_name in METHODS:
        line = []
        for cond in CONDITIONS:
            sub = [r for m, c, _, r, _, _ in rows if m == method_name and c == cond]
            if cond == "noise_only":
                line.append(f"{cond} 幻覺 {sum(r.I > 0 for r in sub)}/{len(sub)}")
            else:
                line.append(f"{cond} {corpus_wer(sub):.1%}")
        print(f"  {method_name:12s} " + "  ".join(line))

    noise_rows = [r for _, c, _, r, _, _ in rows if c == "noise_only"]
    assert all(r.N == 0 and r.S == 0 and r.D == 0 for r in noise_rows)
    clean_baseline = [r for m, c, _, r, _, _ in rows if m == "baseline" and c == "clean"]
    assert corpus_wer(clean_baseline) < 0.5  # 模型確實在辨識（非空輸出、非亂碼）
    empty_after_vad = [(m, c, uid) for m, c, uid, _, _, empty in rows if empty and c != "noise_only"]
    assert not empty_after_vad, f"VAD 把語音句整段刪除：{empty_after_vad}"
