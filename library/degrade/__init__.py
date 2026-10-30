"""產生劣化音訊（實驗條件）。"""

from library.degrade.noise import add_noise, measure_snr
from library.degrade.noise_only import make_noise_only
from library.degrade.registry import (
    apply_condition,
    build_noise_only_clips,
    list_conditions,
    noise_bank,
    pick_noise_env,
    prevent_clipping,
    speech_component,
)
from library.degrade.reverb import add_reverb, estimate_rt60, room_impulse_response
from library.degrade.volume import change_volume

__all__ = [
    "add_noise",
    "add_reverb",
    "apply_condition",
    "build_noise_only_clips",
    "change_volume",
    "estimate_rt60",
    "list_conditions",
    "make_noise_only",
    "measure_snr",
    "noise_bank",
    "pick_noise_env",
    "prevent_clipping",
    "room_impulse_response",
    "speech_component",
]
