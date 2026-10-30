# 決策紀錄

重要決策與理由（模型、資料集、指標的選擇）。初始決策見 `plan.md` 第 2 章「已確定的決策摘要」；
之後的變更依下列格式由新到舊附加。

## 格式

```markdown
## YYYY-MM-DD：<決策標題>
- 決定：
- 理由：
- 替代方案：
- 影響範圍（設定檔 / 模組 / 報告）：
```

---

## 2026-10-30：Parakeet TDT 0.6B v2 暫緩
- 決定：只保留 `library/models/parakeet.py` 空殼與 `configs/models/parakeet-tdt-v2.yaml`（enabled: false）。
- 理由：依賴 NeMo，官方以 Linux 為主，Windows 原生安裝困難。
- 替代方案：WSL2 或 Jetson 上補跑同一份設定。
- 影響範圍：本輪缺少 Transducer 架構比較，報告需註明。
