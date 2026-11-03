# services/llama_cpp/

LLM 服務：llama.cpp 的 `llama-server`（OpenAI 相容 API），給 library/llm 的 LLMClient 連線。
板子上用 transformers 跑 LLM 時，每個 token 的 Python / kernel 啟動負擔在板子 CPU 上很重（Qwen3-0.6B 每秒約 9 個 token）；
同一個模型改用 llama.cpp 約每秒 48 個 token，記憶體也較少（Pipeline Round 1）。
原始碼與編譯結果放在本資料夾的 `llama.cpp/`，不進版控。

## 編譯（Jetson Orin Nano，JetPack 7 / CUDA 13.2，約 45 分鐘）
板子上沒有 cmake，用 uv 從 PyPI 取得（不裝進系統）。只編 Orin 的 sm_87、4 個核心、低優先權（共用板子）。在本資料夾執行：

    git clone --depth 1 --branch v0.8.0 https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp
    export PATH=/usr/local/cuda/bin:$PATH
    uvx --from cmake==4.2.1 cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    nice -n 10 uvx --from cmake==4.2.1 cmake --build build -j4 --target llama-server llama-bench

編譯出的程式以絕對路徑找 build/bin 下的函式庫（RUNPATH）。板子上目前的版本是在別處編譯後搬過來的，已用
`uvx --from patchelf patchelf --set-rpath '$ORIGIN'` 改成「在程式所在資料夾找」；build/ 裡的 CMakeCache.txt 仍是舊路徑，
要重新編譯時先刪掉 build/。

## 模型（GGUF，從 Hugging Face 下載，固定 revision）
| 模型 | repo / 檔案 | revision | 大小 |
|---|---|---|---|
| Qwen3-1.7B Q4_K_M（建議） | unsloth/Qwen3-1.7B-GGUF / Qwen3-1.7B-Q4_K_M.gguf | d7f544eead698dbd1f15126ef60b45a1e1933222 | 1.11 GB |
| Qwen3-0.6B Q8_0 | Qwen/Qwen3-0.6B-GGUF / Qwen3-0.6B-Q8_0.gguf | 23749fefcc72300e3a2ad315e1317431b06b590a | 0.64 GB |

在專案根目錄下載（存到 Hugging Face 快取，印出檔案路徑）：

    uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('unsloth/Qwen3-1.7B-GGUF', 'Qwen3-1.7B-Q4_K_M.gguf', revision='d7f544eead698dbd1f15126ef60b45a1e1933222'))"

## 啟動（在本資料夾）
    llama.cpp/build/bin/llama-server -m <gguf 路徑> -ngl 99 -c 2048 -np 1 --load-mode none --jinja --host 127.0.0.1 --port 8090
- -ngl 99：全部層放 GPU；-c 2048：context 長度（限制 KV cache 記憶體）；-np 1：一個請求槽。
- --load-mode none：不用 mmap（v0.8.0 取代舊的 --no-mmap），避免統一記憶體上模型多佔一份。
- `curl http://127.0.0.1:8090/health` 回 200 就可以連線。
