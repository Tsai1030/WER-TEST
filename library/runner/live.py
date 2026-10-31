"""Round 4：麥克風錄音（07_record_wake.py）與即時手動實測（09_live_demo.py）。

錄音與實測檔案只存在 data/wake/（不進版控、不可公開）。sounddevice 於函式內 import。
"""

import csv
import json
import queue
import time
from collections import deque
from datetime import datetime

import numpy as np

from library.data import read_audio, write_audio
from library.endpoint import SAMPLE_RATE, Endpointer, endpoint_params
from library.kws import KeywordSpotter
from library.models import get_model
from library.runner.wake import WakeConfig, spotter_config
from library.utils.paths import resolve_path

RECORD_LOG = "recordings.csv"
LIVE_LOG = "live_log.jsonl"
_BLOCK = 1600  # 每次從麥克風取 100 ms


def _dbfs(wav: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if len(wav) else 0.0
    return 20 * np.log10(rms) if rms > 0 else -120.0


def beep(sec: float = 0.25, freq: float = 880.0) -> np.ndarray:
    """產生提示音（沒有錄「我在我在」時代替回覆聲）。"""
    t = np.arange(round(sec * SAMPLE_RATE)) / SAMPLE_RATE
    fade = np.minimum(1.0, np.minimum(t, sec - t) / 0.02)
    return (0.3 * np.sin(2 * np.pi * freq * t) * fade).astype(np.float32)


def record_plan(cfg: WakeConfig) -> list[dict]:
    """展開 record.prompts 成逐檔的錄音清單（kind、style、text、hint、sec、path）。

    檔名：<wake_dir>/<kind>/<style>_<nn>.wav；reply 只錄一個檔：<wake_dir>/reply/reply.wav。
    """
    rec = cfg.record
    plan = []
    for p in rec.get("prompts", []):
        n = 1 if p["kind"] == "reply" else int(p.get("n", 1))
        for k in range(1, n + 1):
            name = "reply.wav" if p["kind"] == "reply" else f"{p['style']}_{k:02d}.wav"
            plan.append({"kind": p["kind"], "style": p["style"], "text": p["text"], "hint": p.get("hint", ""),
                         "sec": float(p.get("sec", rec.get("sec", 3.0))), "path": cfg.wake_dir / p["kind"] / name})
    return plan


def run_recorder(cfg: WakeConfig, input_device: int | None = None, redo: bool = False) -> int:
    """依 record_plan() 逐一提示並錄音；已存在的檔案會跳過（redo=True 時重錄）。

    每次錄完顯示音量（dBFS），可選擇保存、重錄或播放。紀錄附加到 <wake_dir>/recordings.csv。

    參數:
        cfg: Round 4 設定。
        input_device: sounddevice 輸入裝置編號（None = 系統預設）。
        redo: 是否重錄已存在的檔案。

    回傳:
        本次保存的檔案數。
    """
    import sounddevice as sd

    plan = [p for p in record_plan(cfg) if redo or not p["path"].is_file()]
    total = len(record_plan(cfg))
    print(f"錄音裝置：{sd.query_devices(input_device, 'input')['name']}；待錄 {len(plan)} / {total} 個檔案")
    print("每次按 Enter 開始錄音；輸入 s 跳過、q 結束。錄完後 Enter 保存、r 重錄、p 播放。\n")
    log_path = cfg.wake_dir / RECORD_LOG
    saved = 0
    for i, p in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {p['kind']} · {p['style']}：請說「{p['text']}」（{p['hint']}，{p['sec']:.0f} 秒）")
        while True:
            cmd = input("  Enter 開始 > ").strip().lower()
            if cmd in ("q", "s"):
                break
            wav = sd.rec(round(p["sec"] * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         device=input_device)
            print("  ● 錄音中…", end="", flush=True)
            sd.wait()
            wav = wav[:, 0]
            level, peak = _dbfs(wav), float(np.max(np.abs(wav)))
            warn = "（太小聲）" if level < -45 else "（可能爆音）" if peak > 0.98 else ""
            print(f"\r  音量 {level:.1f} dBFS、峰值 {peak:.2f}{warn}")
            while (ans := input("  Enter 保存 / r 重錄 / p 播放 > ").strip().lower()) == "p":
                sd.play(wav, SAMPLE_RATE)
                sd.wait()
            if ans == "r":
                continue
            write_audio(p["path"], wav, SAMPLE_RATE)
            new = not log_path.is_file()
            with log_path.open("a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["file", "kind", "style", "text", "sec", "rms_dbfs", "peak", "device", "recorded_at"])
                w.writerow([p["path"].relative_to(cfg.wake_dir).as_posix(), p["kind"], p["style"], p["text"],
                            p["sec"], round(level, 1), round(peak, 3), input_device, datetime.now().isoformat(timespec="seconds")])
            saved += 1
            break
        if cmd == "q":
            break
    print(f"\n本次保存 {saved} 個檔案 → {cfg.wake_dir}")
    return saved


class LiveDemo:
    """即時實測：等待喚醒 → 播放回覆 → 收音 → 講完判斷 → 英文辨識，並顯示各段時間。

    參數:
        cfg: Round 4 設定（使用 live、kws、endpoint.default 區段）。
        overrides: 覆寫 live 區段的欄位（命令列參數），例如 {"threshold": 0.35}。
        kws_only: 只測喚醒（不收音、不辨識），用來實測誤喚醒或喚醒率。
    """

    def __init__(self, cfg: WakeConfig, overrides: dict | None = None, kws_only: bool = False) -> None:
        self.cfg = cfg
        self.live = {**cfg.live, **{k: v for k, v in (overrides or {}).items() if v is not None}}
        self.ep_params = endpoint_params(cfg.endpoint.get("default"))
        self.kws_only = kws_only
        self.out_dir = cfg.wake_dir / "live"

    def _log(self, row: dict) -> None:
        if not self.live.get("save", True):
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / LIVE_LOG).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(self) -> None:
        """執行直到 Ctrl+C。

        例外:
            ImportError: 未安裝 sounddevice / sherpa-onnx（uv sync --extra wake）。
        """
        import sounddevice as sd

        lv = self.live
        kws = KeywordSpotter(spotter_config(self.cfg, lv["kws_model"], float(lv.get("threshold", 0.25))))
        kws.load()
        ep = Endpointer()
        asr = None
        if not self.kws_only:
            ep.load()
            print(f"載入 {lv['asr_model']}…")
            asr = get_model(lv["asr_model"])
            asr.load(lv.get("device", "cpu"), "fp32", int(lv.get("num_threads", 4)))
        reply_path = resolve_path(lv["reply_wav"]) if lv.get("reply_wav") else None
        reply = read_audio(reply_path)[0] if reply_path and reply_path.is_file() else beep()
        device = lv.get("input_device")
        q: queue.Queue = queue.Queue()
        lookback = deque(maxlen=max(1, round(float(lv.get("lookback_ms", 200)) / 100)))
        print(f"喚醒詞模型 {lv['kws_model']}、門檻 {lv.get('threshold', 0.25)}；麥克風："
              f"{sd.query_devices(device, 'input')['name']}")
        print(f"講完判斷：靜音 {self.ep_params['min_silence_ms']:.0f} ms、門檻 {self.ep_params['threshold']}；"
              f"回覆期間{'繼續收音（建議戴耳機）' if lv.get('listen_during_reply', True) else '不收音'}")
        print("說「小咪小咪」喚醒；Ctrl+C 結束。\n")
        n_wake = 0
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=_BLOCK, device=device,
                            callback=lambda data, *_: q.put(data[:, 0].copy())):
            try:
                while True:
                    stream = kws.stream(SAMPLE_RATE)
                    while True:  # 等待喚醒
                        chunk = q.get()
                        lookback.append(chunk)
                        hits = stream.accept(chunk)
                        if hits:
                            break
                    n_wake += 1
                    stamp = datetime.now()
                    print(f"[{stamp:%H:%M:%S}] ✓ 喚醒 #{n_wake}：{hits[0].keyword}")
                    sd.play(reply, SAMPLE_RATE)
                    self._log({"event": "wake", "time": stamp.isoformat(timespec="seconds"), "n": n_wake,
                               "keyword": hits[0].keyword, "kws_model": lv["kws_model"], "threshold": lv.get("threshold")})
                    if self.kws_only:
                        continue
                    if not lv.get("listen_during_reply", True):
                        sd.wait()
                        while not q.empty():  # 丟掉回覆期間收到的聲音
                            q.get()
                        lookback.clear()
                    self._conversation(ep, asr, q, list(lookback), n_wake)
                    lookback.clear()
            except KeyboardInterrupt:
                print(f"\n結束：共喚醒 {n_wake} 次。")
            finally:
                kws.unload()
                ep.unload()
                if asr is not None:
                    asr.unload()

    def _conversation(self, ep: Endpointer, asr: object, q: queue.Queue, first: list[np.ndarray], n_wake: int) -> None:
        """喚醒後的對話：一句一句收音辨識；multi_turn 時直到沒說話逾時才結束。"""
        turn, pending = 0, first
        while True:
            turn += 1
            es = ep.stream(self.ep_params)
            buf: list[np.ndarray] = []
            r = None
            for chunk in pending:
                buf.append(chunk)
                r = es.push(chunk)
            pending = []
            while r is None:
                chunk = q.get()
                buf.append(chunk)
                r = es.push(chunk)
            t_decide = time.perf_counter()
            if r.reason == "no_speech":
                print(f"  （{self.ep_params['no_speech_timeout_s']:.0f} 秒沒有說話，回到等待喚醒）\n")
                self._log({"event": "no_speech", "n": n_wake, "turn": turn})
                return
            audio = np.concatenate(buf)[: round(r.end_sec * SAMPLE_RATE)]
            text = asr.transcribe(audio, SAMPLE_RATE).text
            asr_sec = time.perf_counter() - t_decide
            wait = r.end_sec - (r.speech_end_sec or r.end_sec)
            note = {"max_length": "（超過最長長度，強制結束）", "eof": ""}.get(r.reason, "")
            print(f"  第 {turn} 句{note}：{text}")
            print(f"     說話 {r.speech_end_sec - r.speech_start_sec:.1f} 秒｜講完後等待判定 {wait:.2f} 秒｜"
                  f"辨識 {asr_sec:.2f} 秒｜講完到出字共 {wait + asr_sec:.2f} 秒")
            row = {"event": "turn", "time": datetime.now().isoformat(timespec="seconds"), "n": n_wake, "turn": turn,
                   "reason": r.reason, "text": text, "speech_start": round(r.speech_start_sec, 3),
                   "speech_end": round(r.speech_end_sec, 3), "end_sec": round(r.end_sec, 3),
                   "wait_sec": round(wait, 3), "asr_sec": round(asr_sec, 3), "asr_model": self.live["asr_model"],
                   "endpoint": self.ep_params}
            if self.live.get("save", True):
                name = f"{datetime.now():%Y%m%d_%H%M%S}_w{n_wake:03d}_t{turn}.wav"
                write_audio(self.out_dir / name, audio, SAMPLE_RATE)
                row["file"] = name
            self._log(row)
            if not self.live.get("multi_turn", True):
                print()
                return
