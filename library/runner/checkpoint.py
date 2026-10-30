"""記錄已完成組合，支援中斷續跑（results/<round>/checkpoint.json）。"""

from pathlib import Path


class Checkpoint:
    """已完成的「模型 × 方法 × 條件 × 句子」組合清單，以 utf-8 JSON 存檔。"""

    def __init__(self, path: Path) -> None:
        """參數:
            path: checkpoint.json 路徑；檔案存在時讀入已完成組合。

        例外:
            ValueError: 檔案內容不是合法的 checkpoint。
        """
        raise NotImplementedError

    def is_done(self, model: str, method: str, condition: str, utt_id: str) -> bool:
        """回傳該組合是否已完成。"""
        raise NotImplementedError

    def mark_done(self, model: str, method: str, condition: str, utt_id: str) -> None:
        """標記該組合已完成並立即寫回檔案。"""
        raise NotImplementedError

    def reset(self) -> None:
        """清除所有紀錄（--no-resume 時使用）。"""
        raise NotImplementedError
