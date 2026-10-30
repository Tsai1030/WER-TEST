"""逐行寫入 results.csv、寫 resources.csv（utf-8）。"""

import csv
import math
from dataclasses import asdict
from pathlib import Path
from typing import Self

from library.export.schema import RESOURCE_COLUMNS, RESULT_COLUMNS, ResourceRow, ResultRow


def _cell(value: object) -> object:
    """None 與 NaN 寫成空欄位。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value


class ResultWriter:
    """逐列附加寫入；每寫一列就 flush，確保中斷時資料不遺失。"""

    def __init__(self, path: Path) -> None:
        """參數:
            path: results.csv 路徑；檔案不存在時建立並寫入標題列，存在時附加。

        例外:
            ValueError: 既有檔案的標題列與 ResultRow 欄位不一致。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.is_file() and self.path.stat().st_size > 0
        if exists:
            with self.path.open(encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
            if tuple(header) != RESULT_COLUMNS:
                raise ValueError(f"{self.path} 的標題列與 ResultRow 不一致：{header}")
        self._file = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file, lineterminator="\n")
        if not exists:
            self._writer.writerow(RESULT_COLUMNS)
            self._file.flush()

    def write(self, row: ResultRow) -> None:
        """寫入一列並 flush。None 寫成空欄位。"""
        values = asdict(row)
        self._writer.writerow([_cell(values[c]) for c in RESULT_COLUMNS])
        self._file.flush()

    def close(self) -> None:
        """關閉檔案。"""
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_resources(rows: list[ResourceRow], path: Path) -> None:
    """寫出 resources.csv（覆寫）。

    參數:
        rows: 每個「模型 × 方法」一列。
        path: 輸出路徑。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(RESOURCE_COLUMNS)
        for row in rows:
            values = asdict(row)
            writer.writerow([_cell(values[c]) for c in RESOURCE_COLUMNS])
