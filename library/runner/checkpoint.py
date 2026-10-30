"""記錄已完成組合，支援中斷續跑（results/<round>/checkpoint.json）。

results.csv 才是逐列結果的依據；checkpoint 另外保存已完成的組合鍵、每個「模型 × 方法」的資源量測，
以及各模型的描述，讓續跑時不必重新載入已完成的模型也能寫出完整的 resources.csv 與 meta.json。
"""

import json
import os
from pathlib import Path

from library.export.schema import result_key


class Checkpoint:
    """已完成的「模型 × 方法 × 條件 × 句子」組合清單，以 utf-8 JSON 存檔。"""

    def __init__(self, path: Path) -> None:
        """參數:
            path: checkpoint.json 路徑；檔案存在時讀入已完成組合。

        例外:
            ValueError: 檔案內容不是合法的 checkpoint。
        """
        self.path = Path(path)
        self.done: set[str] = set()
        self.resources: dict[str, dict] = {}  # "model|method" → ResourceRow 欄位
        self.models: dict[str, dict] = {}  # model → describe()
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.done = set(data["done"])
                self.resources = dict(data.get("resources", {}))
                self.models = dict(data.get("models", {}))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"checkpoint 格式錯誤：{self.path}（{exc}）") from exc

    def is_done(self, model: str, method: str, condition: str, utt_id: str) -> bool:
        """回傳該組合是否已完成。"""
        return result_key(model, method, condition, utt_id) in self.done

    def mark_done(self, model: str, method: str, condition: str, utt_id: str) -> None:
        """標記該組合已完成（呼叫 save() 才寫回檔案）。"""
        self.done.add(result_key(model, method, condition, utt_id))

    def save(self) -> None:
        """寫回檔案；先寫暫存檔再取代，避免中斷時留下半個 JSON。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        data = {"done": sorted(self.done), "resources": self.resources, "models": self.models}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def reset(self) -> None:
        """清除所有紀錄（--no-resume 時使用）。"""
        self.done.clear()
        self.resources.clear()
        self.models.clear()
