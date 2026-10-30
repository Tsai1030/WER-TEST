"""固定亂數種子。所有亂數一律透過本模組設定，並將 seed 記錄到 meta.json。"""


def set_seed(seed: int) -> None:
    """同時固定 random、numpy、torch（若已安裝）的亂數種子。

    參數:
        seed: 亂數種子。

    例外:
        ValueError: seed 為負數。
    """
    raise NotImplementedError
