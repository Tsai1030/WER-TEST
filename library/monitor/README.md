# library/monitor/

資源與時間量測。

| 檔案 | 做什麼 |
|---|---|
| memory.py | ResourceMonitor：峰值 RAM（psutil 輪詢）；偵測到 GPU 時另量 GPU |
| timer.py | Timer：毫秒計時 |
| cpu.py | CpuTimer：行程 CPU 秒數與經過秒數（算即時處理的 CPU 佔用）；current_rss_mb()：目前 RSS |

## 範例
    with ResourceMonitor() as mon:
        model.transcribe(wav)
    print(mon.peak_gpu_mb, mon.peak_ram_mb, mon.elapsed_ms)

## 注意
- 本輪為 CPU only：peak_gpu_mb 為 None，只看 RAM；fp32 權重約為 Jetson 部署 q4 的 8 倍，需在 Jetson 實測
- 不同量測方式（torch / pynvml）的數字不可直接比較
