"""統一 log 格式（時間、模組、等級）。"""

import logging


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """設定全域 log 格式；應在 scripts/ 的入口呼叫一次。

    參數:
        level: log 等級（DEBUG / INFO / WARNING / ERROR）。
        log_file: 另外寫入的 log 檔路徑；None 表示只輸出到終端機。檔案以 utf-8 寫入。

    例外:
        ValueError: level 不合法。
    """
    raise NotImplementedError


def get_logger(name: str) -> logging.Logger:
    """取得指定名稱的 logger（通常傳入 __name__）。

    參數:
        name: logger 名稱。

    回傳:
        logging.Logger。
    """
    raise NotImplementedError
