"""產生劣化音訊（實驗條件）。"""

from library.degrade.noise import add_noise
from library.degrade.noise_only import make_noise_only
from library.degrade.registry import apply_condition, list_conditions
from library.degrade.reverb import add_reverb
from library.degrade.volume import change_volume

__all__ = [
    "add_noise",
    "add_reverb",
    "apply_condition",
    "change_volume",
    "list_conditions",
    "make_noise_only",
]
