"""讀取 YAML 設定檔（utf-8）。"""

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """讀取 YAML 檔並回傳 dict。

    參數:
        path: YAML 檔路徑。

    回傳:
        設定內容。

    例外:
        FileNotFoundError: 檔案不存在。
        ValueError: 檔案最外層不是 mapping。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"設定檔不存在：{p}")
    with p.open(encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if not isinstance(content, dict):
        raise ValueError(f"設定檔最外層必須是 mapping：{p}")  # noqa: TRY004  檔案內容錯誤，非呼叫端型別錯誤
    return content
