"""實驗矩陣執行。唯一可以串接其他 library 子模組的地方。"""

from library.runner.bench import BenchResult, bench_model, inferences_per_model
from library.runner.checkpoint import Checkpoint
from library.runner.config import ExperimentConfig, load_config
from library.runner.degrade_set import make_degraded_set
from library.runner.matrix import degraded_path, iter_matrix, split_utterances
from library.runner.runner import OUTPUT_FILES, run_experiment
from library.runner.vad_sweep import run_vad_sweep, summarize_sweep
from library.runner.wake import (
    WakeConfig,
    build_episodes,
    endpoint_variants,
    kws_variants,
    load_wake_config,
    run_endpoint_sim,
    run_kws_negative,
    run_kws_positive,
    summarize_endpoint,
    summarize_kws,
)

__all__ = [
    "OUTPUT_FILES",
    "BenchResult",
    "Checkpoint",
    "ExperimentConfig",
    "WakeConfig",
    "bench_model",
    "build_episodes",
    "degraded_path",
    "endpoint_variants",
    "inferences_per_model",
    "iter_matrix",
    "kws_variants",
    "load_config",
    "load_wake_config",
    "make_degraded_set",
    "run_endpoint_sim",
    "run_experiment",
    "run_kws_negative",
    "run_kws_positive",
    "run_vad_sweep",
    "split_utterances",
    "summarize_endpoint",
    "summarize_kws",
    "summarize_sweep",
]
