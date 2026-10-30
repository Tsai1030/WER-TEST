"""run_experiment() 主流程。"""

from pathlib import Path

from library.runner.config import ExperimentConfig


def run_experiment(cfg: ExperimentConfig, resume: bool = True) -> Path:
    """執行完整實驗矩陣，逐列寫入 results.csv，結束後寫 resources.csv 與 meta.json。

    - 劣化音檔從 cfg.degraded_dir 讀取（預先產生，不即時產生）。
    - 方法回傳空音訊時不呼叫模型，直接記錄空字串 hyp。
    - 每換一個方法重設峰值記憶體統計；每個模型結束後 unload。
    - cfg.save_processed 為 true 時才把方法處理後的音訊存檔。

    參數:
        cfg: 實驗設定。
        resume: 是否依 checkpoint.json 跳過已完成組合。

    回傳:
        結果資料夾路徑（cfg.output_dir）。

    例外:
        FileNotFoundError: manifest 或劣化音檔不存在。
    """
    raise NotImplementedError
