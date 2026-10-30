"""檢查輸出檔是否符合規格（dashboard/data_contract.md）。"""

from pathlib import Path


def validate_results(results_dir: Path) -> list[str]:
    """檢查 results.csv、resources.csv、meta.json：欄位、命名、列數、數值範圍。

    參數:
        results_dir: 結果資料夾（例：results/round1）。

    回傳:
        問題描述清單；空清單代表通過。
    """
    raise NotImplementedError
