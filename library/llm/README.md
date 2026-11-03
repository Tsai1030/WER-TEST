# library/llm/

LLM client：連到另外啟動的 LLM 服務（llama.cpp 的 llama-server，OpenAI 相容串流 API），逐段讀回覆並計時。
只用標準函式庫（http.client），不載入任何模型；LLM 服務的編譯與啟動見 services/llama_cpp/README.md。

| 檔案 | 做什麼 |
|---|---|
| client.py | LLMClient：health()、stream_chat()（逐段 yield 回覆文字）；LLMStats：token 數、第一個 token 時間、總時間、伺服器 timings |

## 範例
    llm = LLMClient("http://127.0.0.1:8090")
    for piece in llm.stream_chat([{"role": "user", "content": "Why is the sky blue?"}], max_tokens=96):
        print(piece, end="")
    print(llm.last_stats.n_tokens, llm.last_stats.tokens_per_s)

## 注意
- temperature 0 = greedy，同一個問題每次回覆相同，方便比較。
- Qwen3 系列加 extra={"chat_template_kwargs": {"enable_thinking": False}} 關掉 thinking，否則會先輸出思考內容。
