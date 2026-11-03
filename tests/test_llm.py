"""LLM client：以本機假的 OpenAI 相容串流伺服器測試（不需要模型）。"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from library.llm import LLMClient

PIECES = ["Hello", " there", ".", " Bye", "."]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:  # 測試時不印存取紀錄
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.last_request = req
        if req["messages"][-1]["content"] == "fail":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for p in PIECES:
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n".encode())
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": {"predicted_n": 7}}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_stream_chat_yields_pieces_and_stats(server) -> None:
    llm = LLMClient(f"http://127.0.0.1:{server.server_address[1]}")
    assert llm.health()
    assert llm.model_id() == "fake-model"
    out = list(llm.stream_chat([{"role": "user", "content": "hi"}], max_tokens=8, extra={"top_k": 1}))
    assert out == PIECES
    assert llm.last_stats.n_tokens == 7  # 以伺服器 timings 為準
    assert llm.last_stats.ttft_s is not None and llm.last_stats.total_s >= llm.last_stats.ttft_s
    assert server.last_request["stream"] is True and server.last_request["max_tokens"] == 8
    assert server.last_request["top_k"] == 1


def test_server_error_and_unreachable(server) -> None:
    llm = LLMClient(f"http://127.0.0.1:{server.server_address[1]}")
    with pytest.raises(RuntimeError, match="500"):
        list(llm.stream_chat([{"role": "user", "content": "fail"}]))
    assert not LLMClient("http://127.0.0.1:9", timeout_s=1).health()  # 連不上回傳 False
    with pytest.raises(ValueError):
        LLMClient("127.0.0.1:8090")
