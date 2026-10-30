"""逐行寫入 results.csv、寫 resources.csv（utf-8）。"""

from pathlib import Path

from library.export.schema import ResourceRow, ResultRow


class ResultWriter:
    """逐列附加寫入；每寫一列就 flush，確保中斷時資料不遺失。"""

    def __init__(self, path: Path) -> None:
        """參數:
            path: results.csv 路徑；檔案不存在時建立並寫入標題列，存在時附加。

        例外:
            ValueError: 既有檔案的標題列與 ResultRow 欄位不一致。
        """
        raise NotImplementedError

    def write(self, row: ResultRow) -> None:
        """寫入一列並 flush。None 寫成空欄位。"""
        raise NotImplementedError

    def close(self) -> None:
        """關閉檔案。"""
        raise NotImplementedError


def write_resources(rows: list[ResourceRow], path: Path) -> None:
    """寫出 resources.csv（覆寫）。

    參數:
        rows: 每個「模型 × 方法」一列。
        path: 輸出路徑。
    """
    raise NotImplementedError
