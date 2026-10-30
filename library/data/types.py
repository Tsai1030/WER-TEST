"""資料結構定義。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Utterance:
    """一句測試語音（或一段純噪音）。

    純噪音片段的 utt_id 為 noise01…noise10，ref_raw、speaker_id、gender 為空字串。
    """

    utt_id: str  # 例：utt001
    wav_path: Path  # clean WAV 路徑
    ref_raw: str  # 原始 reference 文字
    speaker_id: str  # 匿名化 ID（spk01…），不保留 Common Voice client_id
    gender: str  # "female" | "male"
    audio_sec: float
