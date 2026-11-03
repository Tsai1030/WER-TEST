"""LLM client：連到 OpenAI 相容的串流聊天 API（llama.cpp 的 llama-server）。只用標準函式庫。

LLM 本身是另外啟動的服務（見 services/llama_cpp/README.md），這裡只負責送請求、逐段讀回覆與計時。
"""

import http.client
import json
import time
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class LLMStats:
    """最近一次 stream_chat() 的統計（時間皆相對於送出請求）。

    屬性:
        n_tokens: 產生的 token 數（伺服器有回傳 timings 時用它的 predicted_n，否則為收到的片段數）。
        ttft_s: 收到第一段非空白文字的時間；沒有產生文字為 None。
        total_s: 收完回覆的時間。
        timings: 伺服器回傳的 timings（llama-server 才有；沒有為空 dict）。
    """

    n_tokens: int = 0
    ttft_s: float | None = None
    total_s: float = 0.0
    timings: dict = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float | None:
        """整段回覆的產生速度（token / 秒，含第一個 token 之前的時間）。"""
        return self.n_tokens / self.total_s if self.total_s > 0 else None


class LLMClient:
    """OpenAI 相容 API 的串流 client。

    參數:
        url: 伺服器位址，例如 http://127.0.0.1:8090。
        timeout_s: 連線與讀取逾時（秒）。
    """

    def __init__(self, url: str, timeout_s: float = 60.0) -> None:
        u = urllib.parse.urlparse(url)
        if u.scheme != "http" or not u.hostname:
            raise ValueError(f"LLM url 必須是 http://host:port：{url}")
        self.host, self.port = u.hostname, u.port or 80
        self.timeout_s = timeout_s
        self.last_stats = LLMStats()

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)

    def health(self) -> bool:
        """伺服器是否已載入模型、可以接受請求（GET /health 回 200）。連不上時回傳 False。"""
        conn = self._conn()
        try:
            conn.request("GET", "/health")
            return conn.getresponse().status == 200
        except OSError:
            return False
        finally:
            conn.close()

    def model_id(self) -> str | None:
        """伺服器載入的模型名稱（GET /v1/models 的第一個 id）；取不到為 None。"""
        conn = self._conn()
        try:
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            data = json.loads(resp.read()) if resp.status == 200 else {}
            return (data.get("data") or [{}])[0].get("id")
        except (OSError, ValueError):
            return None
        finally:
            conn.close()

    def stream_chat(self, messages: list[dict], max_tokens: int = 96, temperature: float = 0.0,
                    extra: dict | None = None) -> Iterator[str]:
        """送出聊天請求，逐段 yield 回覆文字；讀完後統計放在 last_stats。

        參數:
            messages: [{"role": "system" | "user" | "assistant", "content": str}, ...]。
            max_tokens: 最多產生的 token 數。
            temperature: 取樣溫度；0 = greedy（每次回覆相同）。
            extra: 其他請求欄位，例如 {"chat_template_kwargs": {"enable_thinking": False}}。

        回傳:
            回覆文字片段的 iterator。

        例外:
            RuntimeError: 伺服器回傳非 200。
            OSError: 連不上伺服器或逾時。
        """
        body = {"messages": messages, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
                **(extra or {})}
        stats = LLMStats()
        self.last_stats = stats
        t0 = time.perf_counter()
        conn = self._conn()
        try:
            conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"LLM 伺服器回傳 {resp.status}：{resp.read()[:200]!r}")
            n = 0
            for raw in resp:  # Server-Sent Events：每行 "data: {json}"，最後 "data: [DONE]"
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if "timings" in ev:
                    stats.timings = ev["timings"]
                for choice in ev.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if not piece:
                        continue
                    n += 1
                    if stats.ttft_s is None and piece.strip():
                        stats.ttft_s = time.perf_counter() - t0
                    yield piece
        finally:
            conn.close()
        stats.n_tokens = int(stats.timings.get("predicted_n", n))
        stats.total_s = time.perf_counter() - t0
