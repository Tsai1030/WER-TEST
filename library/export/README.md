# library/export/

輸出結果檔，格式需符合 dashboard/data_contract.md。

| 檔案 | 做什麼 |
|---|---|
| schema.py | ResultRow、ResourceRow 欄位定義 |
| writer.py | ResultWriter（逐列寫入並 flush）、write_resources() |
| meta.py | write_meta()：收集 CPU / GPU、套件版本、git commit、設定 |
| validate.py | validate_results()：檢查欄位、命名、列數、數值範圍 |
