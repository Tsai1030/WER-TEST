# library/utils/

| 檔案 | 做什麼 |
|---|---|
| paths.py | 專案根目錄與 data、results 等路徑集中管理；resolve_path() 將相對路徑解析為相對於專案根目錄 |
| logging.py | 統一 log 格式（時間、模組、等級） |
| seed.py | set_seed()：同時固定 random、numpy、torch |
| yaml_io.py | load_yaml()：以 utf-8 讀取 YAML 設定檔 |
| device.py | resolve_device()：cuda 不可用時退回 cpu 並記錄警告 |
