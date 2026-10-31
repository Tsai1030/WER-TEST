"""解碼方法：不改變音訊，只覆寫模型的解碼參數（由 runner 以 model.set_decoding() 傳給模型）。"""

import numpy as np

from library.methods.base import AudioMethod


class DecodingMethod(AudioMethod):
    """音訊原樣回傳；config 的 decoding（例：{beam_size: 5}）即要覆寫的模型解碼參數。

    參數來自 configs/methods/<name>.yaml（或該輪設定檔的 method_params）。
    不屬於任何模型：模型是否支援由 runner 檢查（ASRModel.supported_decoding）。
    """

    def process(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """原樣回傳 wav（轉為 float32）。"""
        return wav.astype(np.float32, copy=False)

    def decoding(self) -> dict:
        """回傳 config 的 decoding（複本）。"""
        return dict(self.config.get("decoding") or {})


class BeamMethod(DecodingMethod):
    """beam：束搜尋（beam_size 預設 5）。"""

    name = "beam"


class GreedyMethod(DecodingMethod):
    """greedy：貪婪解碼（beam_size 1）。"""

    name = "greedy"
