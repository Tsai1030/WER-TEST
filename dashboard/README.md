# dashboard/

結果網頁讀取 results/<round>/ 的三個檔案，所有彙總由網頁計算。

| 檔案 | 內容 |
|---|---|
| data_contract.md | 三個檔案的欄位、型別與命名規則（必須遵守） |
| sections.md | 網頁 5 個分頁、各區塊的定義與計算方式 |

## 更新流程
1. uv run python scripts/04_validate_results.py results/round1
2. 將三個檔案交給 Claude 更新網頁
3. 對照 05_quick_report.py 的摘要確認數字一致
