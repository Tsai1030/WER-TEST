# stt test（STT WER 測試專案）

找出 STT 錯誤來源並驗證降低 WER 的方法，同時記錄記憶體與延遲代價。

## 快速開始
1. 建環境：見 docs/plan.md 第 11 章（uv sync --extra audio --extra models --extra wake）
2. uv run python scripts/check_env.py
3. uv run python scripts/01_prepare_data.py --config configs/round1.yaml
4. uv run python scripts/02_make_degraded.py --config configs/round1.yaml
5. uv run python scripts/03_run_matrix.py --config configs/round1.yaml
6. uv run python scripts/04_validate_results.py results/round1
7. uv run python scripts/05_quick_report.py results/round1

## 資料夾導覽
| 資料夾 | 用途 |
|---|---|
| configs/ | 所有實驗參數 |
| library/ | 核心函式庫，可 import 呼叫 |
| scripts/ | 命令列入口 |
| data/ | 資料（不進版控） |
| results/ | 實驗輸出 |
| dashboard/ | 結果網頁的資料規格 |
| tests/ | 測試 |
| docs/ | 計畫書、決策紀錄、實驗紀錄 |
