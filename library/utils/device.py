"""裝置選擇：程式不可假設有 GPU。"""

import logging

log = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """依設定檔的 device 決定實際使用的裝置。

    要求 cuda 但不可用（或未安裝 torch）時退回 cpu 並記錄警告。

    參數:
        device: "cpu"、"cuda" 或 "cuda:N"。

    回傳:
        實際使用的裝置字串。

    例外:
        ValueError: device 不是 cpu / cuda。
    """
    if device == "cpu":
        return "cpu"
    if not device.startswith("cuda"):
        raise ValueError(f"不支援的 device：{device}（只接受 cpu / cuda）")
    try:
        import torch
    except ImportError:
        log.warning("要求 %s 但未安裝 torch，改用 cpu", device)
        return "cpu"
    if not torch.cuda.is_available():
        log.warning("要求 %s 但 CUDA 不可用，改用 cpu", device)
        return "cpu"
    return device
