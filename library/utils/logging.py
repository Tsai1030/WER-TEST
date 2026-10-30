"""統一 log 格式（時間、模組、等級）。"""

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """設定全域 log 格式；應在 scripts/ 的入口呼叫一次。

    參數:
        level: log 等級（DEBUG / INFO / WARNING / ERROR）。
        log_file: 另外寫入的 log 檔路徑；None 表示只輸出到終端機。檔案以 utf-8 寫入。

    例外:
        ValueError: level 不合法。
    """
    levels = logging.getLevelNamesMapping()
    if level.upper() not in levels:
        raise ValueError(f"不合法的 log 等級：{level}")
    numeric = levels[level.upper()]
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=numeric, format=_FORMAT, datefmt=_DATEFMT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """取得指定名稱的 logger（通常傳入 __name__）。

    參數:
        name: logger 名稱。

    回傳:
        logging.Logger。
    """
    return logging.getLogger(name)
