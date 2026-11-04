"""整段語音助理的手動實測（scripts/10_assistant_live.py）：麥克風 → 講完判斷 → STT → LLM（串流）→ TTS → 喇叭。

LLM（llama-server）與 TTS（services/tts_server）是另外啟動的服務，這裡只透過 library.llm / library.tts 的 client 連線。
LLM 每產生完一句就交給 TTS，第一句合成好就開始播放；每一輪印出並記錄各段時間。
錄音、回覆音檔與紀錄只存在 data/assistant/（不進版控、不可公開）。sounddevice、soxr 於函式內 import。
音效卡不支援 16 kHz 收音（或 TTS 的取樣率播放）時，改用裝置預設取樣率開啟，收音即時轉成 16 kHz、播放前轉成裝置取樣率。
"""

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from library.data import write_audio
from library.endpoint import SAMPLE_RATE, Endpointer, endpoint_params
from library.llm import LLMClient
from library.models import get_model, list_models
from library.tts import TTSClient
from library.utils.device import resolve_device
from library.utils.logging import get_logger
from library.utils.paths import resolve_path
from library.utils.yaml_io import load_yaml

log = get_logger(__name__)
LIVE_LOG = "live_log.jsonl"
_BLOCK = 1600  # 每次從麥克風取 100 ms
SENT_END = re.compile(r"(?<=[.!?])\s+")
_SECTIONS = ("asr", "endpoint", "llm", "tts", "live")


@dataclass
class AssistantConfig:
    """整段語音助理設定（configs/assistant.yaml）。各區段保留為 dict，欄位說明見設定檔註解。"""

    name: str
    out_dir: Path
    asr: dict
    endpoint: dict
    llm: dict
    tts: dict
    live: dict = field(default_factory=dict)


def load_assistant_config(path: str | Path, overrides: dict | None = None) -> AssistantConfig:
    """讀取並驗證設定；overrides 為 {區段: {欄位: 值}}，值為 None 的欄位不覆寫（命令列沒給的參數）。

    例外:
        FileNotFoundError: 設定檔不存在。
        ValueError: 缺少必要區段或欄位、STT 模型未註冊、endpoint 參數不合法。
    """
    raw = load_yaml(resolve_path(path))
    for key in ("name", "paths", "asr", "llm", "tts"):
        if key not in raw:
            raise ValueError(f"設定檔缺少 {key}")
    sec = {k: dict(raw.get(k) or {}) for k in _SECTIONS}
    for k, v in (overrides or {}).items():
        if k not in _SECTIONS:
            raise ValueError(f"未知的設定區段：{k}")
        sec[k].update({f: x for f, x in v.items() if x is not None})
    if sec["asr"].get("model") not in list_models():
        raise ValueError(f"asr.model 必須是 {list_models()} 之一：{sec['asr'].get('model')}")
    if not sec["llm"].get("url"):
        raise ValueError("llm.url 不可為空")
    if not sec["tts"].get("port"):
        raise ValueError("tts.port 不可為空")
    endpoint_params(sec["endpoint"])  # 驗證參數
    return AssistantConfig(name=raw["name"], out_dir=resolve_path(raw["paths"].get("out_dir", "data/assistant")), **sec)


def split_sentences(buf: str) -> tuple[list[str], str]:
    """把串流文字中已經完整的句子（以 . ! ? 加空白結尾）切出來。

    參數:
        buf: 目前累積、尚未送出的文字。

    回傳:
        (完整句子清單, 剩下還沒結束的文字)。
    """
    parts = SENT_END.split(buf)
    return [p.strip() for p in parts[:-1] if p.strip()], parts[-1]


def turn_timings(t: dict) -> dict:
    """由一輪各事件的時間點（time.perf_counter 秒）算出各段秒數。

    參數:
        t: speech_end（最後一個語音幀結束，由講完判斷回推）、decided（判定講完 = STT 開始）、stt_done、
            llm_start、llm_first_sent、llm_done、first_synth、all_synth、first_play、play_end；
            可有 llm_first_token（沒有產生文字時省略）。

    回傳:
        wait_s（講完到判定）、stt_s、llm_ttft_s、llm_first_sent_s、llm_total_s；
        first_audio_s、all_synth_s、play_end_s（從講完起算）；
        stt_to_first_audio_s、stt_to_all_synth_s、stt_to_play_end_s（從 STT 開始起算）。皆四捨五入到毫秒。
    """
    r = lambda a, b: round(t[a] - t[b], 3) if a in t and b in t else None
    return {
        "wait_s": r("decided", "speech_end"), "stt_s": r("stt_done", "decided"),
        "llm_ttft_s": r("llm_first_token", "llm_start"), "llm_first_sent_s": r("llm_first_sent", "llm_start"),
        "llm_total_s": r("llm_done", "llm_start"),
        "first_audio_s": r("first_play", "speech_end"), "all_synth_s": r("all_synth", "speech_end"),
        "play_end_s": r("play_end", "speech_end"),
        "stt_to_first_audio_s": r("first_play", "decided"), "stt_to_all_synth_s": r("all_synth", "decided"),
        "stt_to_play_end_s": r("play_end", "decided"),
    }


def device_rate(sd: object, device: int | str | None, kind: str, want: int) -> int:
    """選擇音訊裝置的取樣率：支援 want 就用 want，否則用裝置預設取樣率。

    ALSA 的 hw 裝置（例如 USB 音效卡）不會自動轉換取樣率，常只支援 44.1 / 48 kHz。

    參數:
        sd: sounddevice 模組。
        device: 裝置編號或名稱；None = 系統預設。
        kind: "input" 或 "output"。
        want: 想要的取樣率。

    回傳:
        要用來開啟裝置的取樣率。
    """
    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    try:
        check(device=device, samplerate=want, channels=1)
        return want
    except sd.PortAudioError:
        return int(sd.query_devices(device, kind)["default_samplerate"])


class LiveAssistant:
    """即時對話：持續收音 → 講完判斷 → STT → LLM 串流 → 一句一句 TTS 並播放，印出並記錄各段時間。

    參數:
        cfg: 整段語音助理設定（load_assistant_config）。
    """

    def __init__(self, cfg: AssistantConfig) -> None:
        self.cfg = cfg
        self.ep_params = endpoint_params(cfg.endpoint)
        self.history: list[dict] = []
        self.llm_model: str | None = None
        self._out_rates: dict[int, int] = {}  # TTS 取樣率 → 喇叭取樣率

    def _log(self, row: dict) -> None:
        if not self.cfg.live.get("save", True):
            return
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.cfg.out_dir / LIVE_LOG).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _connect(self) -> tuple[LLMClient, TTSClient, dict]:
        llm = LLMClient(self.cfg.llm["url"])
        if not llm.health():
            raise RuntimeError(f"連不上 LLM 服務 {self.cfg.llm['url']}：請先啟動 llama-server（services/llama_cpp/README.md）")
        tts = TTSClient(self.cfg.tts.get("host", "127.0.0.1"), int(self.cfg.tts["port"]))
        try:
            info = tts.ping()
        except OSError as e:
            raise RuntimeError(f"連不上 TTS 服務 {tts.host}:{tts.port}：請先啟動 services/tts_server（{e}）") from e
        return llm, tts, info

    def run(self) -> None:
        """連線服務、載入 STT 與 VAD、暖機後開始對話，直到 Ctrl+C。

        例外:
            RuntimeError: 連不上 LLM 或 TTS 服務。
            ImportError: 未安裝 sounddevice / silero-vad（uv sync --extra wake --extra audio）。
        """
        import sounddevice as sd
        import soxr

        cfg, lv = self.cfg, self.cfg.live
        llm, tts, tts_info = self._connect()
        self.llm_model = llm.model_id()
        print(f"LLM：{self.llm_model}（{cfg.llm['url']}）；TTS：{tts_info.get('model')}（{tts_info.get('device')}，"
              f"補齊 {tts_info.get('pad')}，聲音 {cfg.tts.get('voice', 'F1')}）")
        ep = Endpointer()
        ep.load()
        asr = get_model(cfg.asr["model"])
        device, dtype = resolve_device(cfg.asr.get("device", "cpu")), cfg.asr.get("dtype", "fp32")
        if device == "cpu" and dtype != "fp32":
            log.warning("STT 在 cpu 上執行，dtype 由 %s 改為 fp32", dtype)
            dtype = "fp32"
        print(f"載入 {cfg.asr['model']}（{device}、{dtype}）…")
        asr.load(device, dtype, int(cfg.asr.get("num_threads", 4)))
        asr.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)  # 暖機：第一次推論要初始化 GPU
        for _ in llm.stream_chat([{"role": "user", "content": "Hi"}], max_tokens=4, extra=cfg.llm.get("extra")):
            pass
        q: queue.Queue = queue.Queue()
        mic_sr = device_rate(sd, lv.get("input_device"), "input", SAMPLE_RATE)
        rs = None if mic_sr == SAMPLE_RATE else soxr.ResampleStream(mic_sr, SAMPLE_RATE, 1, dtype="float32")
        print(f"麥克風：{sd.query_devices(lv.get('input_device'), 'input')['name']}"
              f"{f'（{mic_sr} Hz 收音，轉成 {SAMPLE_RATE} Hz）' if rs else ''}；"
              f"喇叭：{sd.query_devices(lv.get('output_device'), 'output')['name']}")
        print(f"講完判斷：靜音 {self.ep_params['min_silence_ms']:.0f} ms；"
              f"回覆期間{'繼續收音（建議戴耳機）' if lv.get('listen_during_reply') else '不收音'}。Ctrl+C 結束。\n")
        n = 0
        with sd.InputStream(samplerate=mic_sr, channels=1, dtype="float32", blocksize=round(_BLOCK * mic_sr / SAMPLE_RATE),
                            device=lv.get("input_device"), callback=lambda data, *_: q.put(data[:, 0].copy())):
            try:
                while True:
                    es = ep.stream(self.ep_params)
                    buf: list[np.ndarray] = []
                    r = None
                    while r is None:
                        chunk = q.get()
                        if rs is not None:
                            chunk = rs.resample_chunk(chunk)
                        buf.append(chunk)
                        r = es.push(chunk)
                    if r.reason == "no_speech":
                        continue
                    t_decided = time.perf_counter()
                    n += 1
                    audio = np.concatenate(buf)[: round(r.end_sec * SAMPLE_RATE)]
                    self._turn(n, audio, r, t_decided, asr, llm, tts, sd)
                    if not lv.get("listen_during_reply"):
                        while not q.empty():  # 丟掉處理與回覆期間收到的聲音（外放時含回覆聲）
                            q.get()
            except KeyboardInterrupt:
                print(f"\n結束：共 {n} 輪。")
            finally:
                ep.unload()
                asr.unload()
                tts.close()

    def _turn(self, n: int, audio: np.ndarray, r: object, t_decided: float, asr: object, llm: LLMClient,
              tts: TTSClient, sd: object) -> None:
        """一輪：STT → LLM 串流（一句一句交給 TTS）→ 依序播放；印出並記錄各段時間。"""
        cfg, lc, tc = self.cfg, self.cfg.llm, self.cfg.tts
        t = {"decided": t_decided, "speech_end": t_decided - (r.end_sec - r.speech_end_sec)}
        text = asr.transcribe(audio, SAMPLE_RATE).text.strip()
        t["stt_done"] = time.perf_counter()
        print(f"[{datetime.now():%H:%M:%S}] 你：{text or '（沒有辨識出文字）'}")
        if not text:
            return
        keep = 2 * int(lc.get("history_turns", 0))
        messages = ([{"role": "system", "content": lc["system"]}] if lc.get("system") else []) + \
            (self.history[-keep:] if keep else []) + [{"role": "user", "content": text}]

        synth_q: queue.Queue = queue.Queue()
        play_q: queue.Queue = queue.Queue()
        synths: list[tuple[str, object, float]] = []  # (句子, TTSResult, 合成完成時間)
        plays: list[tuple[float, float]] = []
        errors: list[Exception] = []

        def _synth() -> None:
            try:
                while (s := synth_q.get()) is not None:
                    res = tts.synthesize(s, tc.get("voice", "F1"), tc.get("lang", "en"))
                    synths.append((s, res, time.perf_counter()))
                    play_q.put(res)
            except Exception as e:  # noqa: BLE001  讓主執行緒顯示錯誤，不卡住
                errors.append(e)
            finally:
                play_q.put(None)

        def _play() -> None:
            import soxr

            out_dev = cfg.live.get("output_device")
            while (res := play_q.get()) is not None:
                sr = self._out_rates.setdefault(res.sample_rate, device_rate(sd, out_dev, "output", res.sample_rate))
                wav = res.wav if sr == res.sample_rate else soxr.resample(res.wav, res.sample_rate, sr).astype(np.float32)
                start = time.perf_counter()
                sd.play(wav, sr, device=out_dev)
                sd.wait()
                plays.append((start, time.perf_counter()))

        workers = [threading.Thread(target=_synth), threading.Thread(target=_play)]
        for w in workers:
            w.start()
        t["llm_start"] = time.perf_counter()
        buf = reply = ""
        try:
            for piece in llm.stream_chat(messages, int(lc.get("max_tokens", 96)), float(lc.get("temperature", 0.0)),
                                         lc.get("extra")):
                reply += piece
                sents, buf = split_sentences(buf + piece)
                for s in sents:
                    t.setdefault("llm_first_sent", time.perf_counter())
                    synth_q.put(s)
            if buf.strip():
                t.setdefault("llm_first_sent", time.perf_counter())
                synth_q.put(buf.strip())
        finally:
            t["llm_done"] = time.perf_counter()
            synth_q.put(None)
            for w in workers:
                w.join()
        reply = reply.strip()
        print(f"           助理：{reply}")
        if errors or not synths:
            print(f"  （TTS 失敗：{errors[0] if errors else '沒有回覆文字'}）\n")
            return
        stats = llm.last_stats
        if stats.ttft_s is not None:
            t["llm_first_token"] = t["llm_start"] + stats.ttft_s
        t.update(first_synth=synths[0][2], all_synth=synths[-1][2], first_play=plays[0][0], play_end=plays[-1][1])
        tm = turn_timings(t)
        reply_audio = sum(res.audio_s for _, res, _ in synths)
        tps = stats.tokens_per_s
        print(f"     講完判定 {tm['wait_s']:.2f} s｜STT {tm['stt_s']:.2f} s｜LLM 第一個 token {tm['llm_ttft_s'] or 0:.2f} s、"
              f"第一句 {tm['llm_first_sent_s']:.2f} s（{stats.n_tokens} token，{tps or 0:.1f} token/s）｜"
              f"TTS 第一句 {synths[0][1].synth_s:.2f} s")
        print(f"     講完 → 開始出聲 {tm['first_audio_s']:.2f} s｜全部合成完 {tm['all_synth_s']:.2f} s｜"
              f"全部播完 {tm['play_end_s']:.2f} s（回覆 {reply_audio:.1f} s）")
        print(f"     STT 開始 → 開始出聲 {tm['stt_to_first_audio_s']:.2f} s｜全部合成完 {tm['stt_to_all_synth_s']:.2f} s｜"
              f"全部播完 {tm['stt_to_play_end_s']:.2f} s\n")
        row = {"time": datetime.now().isoformat(timespec="seconds"), "turn": n, "text": text, "reply": reply,
               "reason": r.reason, "speech_sec": round(r.speech_end_sec - r.speech_start_sec, 3), **tm,
               "llm_tokens": stats.n_tokens, "llm_tok_per_s": round(tps, 1) if tps else None,
               "tts_first_synth_s": synths[0][1].synth_s, "tts_synth_total_s": round(sum(res.synth_s for _, res, _ in synths), 3),
               "reply_audio_s": round(reply_audio, 3), "n_chunks": len(synths), "asr": cfg.asr, "llm_model": self.llm_model,
               "tts": tc, "endpoint": self.ep_params}
        if cfg.live.get("save", True):
            stamp = f"{datetime.now():%Y%m%d_%H%M%S}_t{n:03d}"
            write_audio(cfg.out_dir / f"{stamp}_user.wav", audio, SAMPLE_RATE)
            write_audio(cfg.out_dir / f"{stamp}_reply.wav", np.concatenate([res.wav for _, res, _ in synths]),
                        synths[0][1].sample_rate)
            row["files"] = [f"{stamp}_user.wav", f"{stamp}_reply.wav"]
        self._log(row)
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
