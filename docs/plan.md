# STT WER 錯誤分析與改善 — 計畫規格書

| 項目 | 內容 |
|---|---|
| 版本 | v0.5（第一輪：英文；Windows 原生、CPU only、uv 環境、Parakeet 暫緩；M0 後：Whisper 改 small、Qwen3-ASR 改 transformers 原生版、降噪改 noisereduce、Jetson 部署預計 q4） |
| 日期 | 2026-10-30 |
| 負責人 | Tsai |
| 目標平台 | 開發：Windows 原生、**CPU only**（Intel i3-10110U、RAM 15.8GB、無獨立顯卡）；部署驗證：Jetson Orin Nano 8GB |
| 專案路徑 | `D:\stt test` |
| 狀態 | 規格確認中 |

---

## 目錄

1. [專案目標與範圍](#1-專案目標與範圍)
2. [已確定的決策摘要](#2-已確定的決策摘要)
3. [系統架構](#3-系統架構)
4. [資料夾結構](#4-資料夾結構)
5. [Library 公開 API 規格](#5-library-公開-api-規格)
6. [資料格式規格](#6-資料格式規格)
7. [設定檔規格](#7-設定檔規格)
8. [實驗設計](#8-實驗設計)
9. [指標定義](#9-指標定義)
10. [結果網頁規格](#10-結果網頁規格)
11. [環境建置](#11-環境建置)
12. [里程碑與驗收標準](#12-里程碑與驗收標準)
13. [風險與對策](#13-風險與對策)
14. [程式撰寫規範](#14-程式撰寫規範)
15. [附錄 A：各資料夾 README 內容](#附錄-a各資料夾-readme-內容)
16. [附錄 B：呼叫範例](#附錄-b呼叫範例)

---

## 1. 專案目標與範圍

### 1.1 目標

找出 STT（語音轉文字）在「聲音 → STT → 評分」流程中造成 WER 升高的原因，並用**小規模、可重現**的實驗驗證哪些方法能降低 WER，同時記錄每個方法的**記憶體與延遲代價**，作為之後部署到 Jetson Orin Nano 8GB 的依據。

核心問題：

1. 錯誤發生在哪一段？（音訊品質 / 模型辨識 / 解碼 / 評分本身）
2. 哪個方法對哪種模型架構有效？
3. 改善幅度是否顯著？代價（記憶體、延遲）是否值得？

### 1.2 本輪範圍（Round 1）

| 包含 | 不包含（之後再做） |
|---|---|
| 英文、WER 指標 | 中文 / 中英夾雜（CER / MER） |
| 3 個模型（Granite、Whisper、Qwen3-ASR） | 模型微調（LoRA 等）；**Parakeet TDT 0.6B v2（暫緩）** |
| Exp0 正規化、Exp1 VAD、Exp2 降噪 | Exp3 熱詞、Exp4 解碼參數、Exp5 LLM 修正 |
| Windows 原生環境上驗證 | Jetson 上的完整測試（只做抽樣複驗） |
| 資源與延遲記錄 | 速度優化、TensorRT 轉換 |

### 1.3 設計原則

- **一次只改一個變因**：方法、條件、模型三個維度各自獨立，結果才能歸因。
- **小步驗證**：50 句、固定 seed，先確認方向，不追求一次做到最好。
- **可重現**：所有參數寫在設定檔，所有結果附帶 `meta.json`。
- **可擴充**：新增模型或方法只需新增一個檔案並註冊，不需修改執行流程。

---

## 2. 已確定的決策摘要

| 項目 | 決定 | 理由 |
|---|---|---|
| 語言 / 指標 | 英文 / WER（+ S/D/I 拆解） | 先單一語言，指標單純 |
| 資料集 | Common Voice Scripted Speech 26.0 American English（Female + Male）test split | CC0 授權、已驗證、說話者多、可分男女 |
| 抽樣 | 女 25 + 男 25，每位說話者 1 句，3–8 秒、≥ 5 詞、down_votes = 0，seed 42 | 避免單一說話者主導；只取無反對票的句子以提高標註品質 |
| 純噪音組 | 10 段不含語音的噪音音檔 | 量測幻覺率 |
| 模型（英文專用） | Granite Speech 5.0 470M TurboCTC（Apache 版） | CTC 架構 |
| **暫緩模型** | Parakeet TDT 0.6B v2（Transducer） | 依賴 NeMo，官方以 Linux 為主，Windows 原生安裝困難；介面與設定檔先保留，待有 Linux 環境（WSL2 或 Jetson）再加入 |
| 模型（多語） | Whisper small、Qwen3-ASR-0.6B | Encoder-decoder 與 Audio-LLM 架構，之後可延伸中文；Whisper 原為 large-v3-turbo，M0 實測 CPU 上 24 s/句，改 small（11.6 s/句）以保留完整 8 個條件 |
| 備案模型 | SenseVoice-Small | 若 Qwen3-ASR 部署困難 |
| 精度 | fp32（CPU） | CPU 上 int8 / q4 不會加速（M0 實測 int8 反而較慢），且三個模型的 q4 需各自不同推論引擎；Jetson 部署預計為 **q4**，屆時以部署版本複驗方法改善幅度是否一致（見 11.5） |
| 開發環境 | Windows 原生、CPU only，以 **uv** 建立獨立環境，專案路徑 `D:\stt test` | 目前無法使用 Ubuntu、無 NVIDIA 顯卡；程式維持跨平台並支援 `device` 切換，之後可直接搬到 Jetson 使用 GPU |
| 噪音源 | DEMAND（16kHz 版，挑選數個環境） | MUSAN 完整檔約 11GB，D 槽空間不足；DEMAND 單一環境僅數十 MB |
| 結果呈現 | 互動式結果網頁（Dashboard），讀取三個輸出檔 | 見第 10 章 |

**已知限制（報告中需註明）**

- Common Voice 可能出現在模型訓練資料中（資料污染），baseline WER 可能偏低；本研究看的是**方法開/關的差異**，影響較小。
- Common Voice 為朗讀語音，沒有贅詞與對話特性；之後可用 MSLT 補充。
- 說話者為美式口音，與未來使用者（台灣口音英文）有落差。
- 開發機為低功耗 CPU：**延遲數字與 Jetson GPU 完全不同，只能比較方法之間的相對差異**；記憶體只量 RAM，且 fp32 權重約為 q4 的 8 倍，不能直接當作 Nano 用量。
- 開發機用 fp32 量方法效果，部署為 q4：量化可能改變模型對噪音的敏感度，方法的改善幅度需在 Jetson 上以 q4 版本複驗。
- Parakeet 暫緩，本輪缺少 Transducer 架構，「哪類方法對哪類架構有效」的比較只涵蓋 CTC、Encoder-decoder、Audio-LLM 三類。
- Common Voice 條款禁止重新散佈資料集與辨識說話者：音檔不可放到公開位置。

---

## 3. 系統架構

### 3.1 處理流程

```
 ┌──────────────┐
 │ 1. 資料準備   │  Common Voice test → 抽樣 50 句 → MP3 轉 16kHz mono WAV → manifest.csv
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │ 2. 劣化產生   │  clean / snr20 / snr10 / snr05 / snr00 / reverb / lowvol  + noise_only
 └──────┬───────┘
        ▼
 ┌──────────────┐      ┌────────────────┐
 │ 3. 方法層     │ ◀─── │                │
 │  （前處理）   │      │  資源監測        │  峰值 RAM（有 GPU 時含 GPU）、每句延遲、載入時間
 └──────┬───────┘      │                │
        ▼              │                │
 ┌──────────────┐      │                │
 │ 4. STT 模型   │ ◀─── │                │
 └──────┬───────┘      └────────────────┘
        ▼
 ┌──────────────┐
 │ 5. 文字正規化 │  ref 與 hyp 用同一個 normalizer
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │ 6. 對齊評分   │  逐句 WER、S / D / I
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │ 7. 輸出       │  results.csv / resources.csv / meta.json → 結果網頁
 └──────────────┘
```

- 第 3、4 層是**實驗變因**（可替換）；其餘為固定流程。
- 每個「模型 × 條件 × 方法」組合各跑一次。

### 3.2 實驗插入點

| 層 | 本輪 | 之後 / 備選 |
|---|---|---|
| 音訊前處理 | Exp1 VAD + 音量正規化、Exp2 降噪 × SNR | WPE 去殘響 |
| 模型解碼 | （固定參數） | Exp3 熱詞 / prompt、Exp4 解碼參數、CTC + KenLM |
| 文字後處理 | Exp0 正規化（建立 baseline） | Exp5 LLM N-best 修正 |

### 3.3 執行順序（迴圈設計）

為了減少模型重複載入，迴圈順序為：

```
for model in models:              # 最外層：模型只載入一次
    load(model)
    for method in methods:        # 方法：每換一個方法重設峰值記憶體統計
        for condition in conditions:
            for utt in utterances:
                wav   = load_audio(condition, utt)
                wav2  = method.process(wav)
                hyp   = model.transcribe(wav2)
                write_row(...)
    unload(model)
```

- 劣化音檔**預先產生並存檔**（第 2 步），不在迴圈內即時產生，確保每個模型聽到的是完全相同的音訊。
- 方法處理後的音訊**不存檔**（只在記憶體中），避免硬碟爆量；需要除錯時可用 `--save-processed` 開啟。

---

## 4. 資料夾結構

```
D:\stt test\                    專案根目錄
├── README.md                    專案入口：目的、快速開始、資料夾導覽
├── pyproject.toml               套件設定與相依套件（分組：core / models / dev）
├── .gitignore                   排除 data/、results/ 中的大型檔案
│
├── configs/                     所有實驗參數（不寫死在程式中）
│   ├── README.md
│   ├── round1.yaml              第一輪實驗總設定（指定模型、方法、條件）
│   ├── conditions.yaml          劣化條件定義（SNR、殘響、音量）
│   ├── models/
│   │   ├── granite-470m.yaml
│   │   ├── parakeet-tdt-v2.yaml  （暫緩，enabled: false）
│   │   ├── whisper-small.yaml
│   │   └── qwen3-asr-0.6b.yaml
│   └── methods/
│       ├── vad.yaml
│       ├── denoise.yaml
│       └── loudnorm.yaml
│
├── library/                     核心函式庫（可被他人 import 呼叫）
│   ├── README.md
│   ├── __init__.py              對外的高階 API（transcribe_file、evaluate 等）
│   │
│   ├── data/                    資料集讀取、抽樣、轉檔、manifest
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── types.py             Utterance 等資料結構
│   │   ├── commonvoice.py       讀取 Common Voice TSV
│   │   ├── sampler.py           依規則抽樣（性別、時長、詞數、每人一句）
│   │   ├── audio_io.py          讀寫音檔、MP3 → 16kHz mono WAV
│   │   └── manifest.py          產生 / 讀取 / 驗證 manifest.csv
│   │
│   ├── degrade/                 產生劣化音訊（實驗條件）
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── noise.py             依指定 SNR 疊加噪音
│   │   ├── reverb.py            模擬房間殘響（pyroomacoustics）
│   │   ├── volume.py            音量增減
│   │   ├── noise_only.py        切出純噪音片段（幻覺測試用）
│   │   └── registry.py          條件名稱 → 處理函式的對應表
│   │
│   ├── methods/                 音訊前處理方法（實驗變因）
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── base.py              AudioMethod 抽象類別
│   │   ├── registry.py          方法註冊與 get_method()（支援 "vad+denoise" 組合）
│   │   ├── chain.py             MethodChain：依序串接多個方法
│   │   ├── identity.py          baseline（不處理）
│   │   ├── vad.py               Silero VAD：去除非語音段
│   │   ├── denoise.py           降噪（noisereduce；原規劃 DeepFilterNet）
│   │   └── loudnorm.py          響度 / RMS 正規化
│   │
│   ├── models/                  STT 模型包裝（統一介面）
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── base.py              ASRModel 抽象類別、Transcript 資料結構
│   │   ├── registry.py          模型註冊與 get_model()
│   │   ├── granite.py           Granite Speech 5.0 470M TurboCTC
│   │   ├── parakeet.py          Parakeet TDT 0.6B v2（暫緩：保留空殼，不實作）
│   │   ├── whisper.py           Whisper small
│   │   └── qwen3_asr.py         Qwen3-ASR-0.6B
│   │
│   ├── textnorm/                文字正規化
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── english.py           英文正規化（Whisper EnglishTextNormalizer + 贅詞移除）
│   │   └── registry.py          get_normalizer()，預留中文版
│   │
│   ├── scoring/                 對齊與評分
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── align.py             Levenshtein 對齊，輸出 S/D/I 與對齊序列
│   │   ├── wer.py               逐句 / 整體 WER
│   │   └── stats.py             配對 bootstrap 信賴區間、變好/變差統計
│   │
│   ├── monitor/                 資源與時間量測
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── memory.py            峰值 RAM（psutil）；有 GPU 時另量 GPU
│   │   └── timer.py             延遲計時
│   │
│   ├── runner/                  實驗矩陣執行
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── config.py            讀取並驗證 YAML 設定
│   │   ├── matrix.py            展開「模型 × 方法 × 條件 × 句子」組合
│   │   ├── checkpoint.py        記錄已完成組合，支援中斷續跑
│   │   └── runner.py            run_experiment() 主流程
│   │
│   ├── export/                  輸出結果檔
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── schema.py            三個輸出檔的欄位定義
│   │   ├── writer.py            逐行寫入 results.csv、寫 resources.csv
│   │   ├── meta.py              收集環境資訊並寫 meta.json
│   │   └── validate.py          檢查輸出檔是否符合規格
│   │
│   └── utils/                   共用工具
│       ├── README.md
│       ├── __init__.py
│       ├── paths.py             專案路徑集中管理
│       ├── logging.py           統一 log 格式
│       └── seed.py              固定亂數種子
│
├── scripts/                     命令列入口（薄包裝，邏輯都在 library）
│   ├── README.md
│   ├── check_env.py             檢查 Python、套件版本、可用裝置（CPU / CUDA）、模型能否載入
│   ├── 01_prepare_data.py       抽樣 + 轉檔 + 產生 manifest
│   ├── 02_make_degraded.py      產生所有劣化條件音檔
│   ├── 03_run_matrix.py         執行實驗矩陣
│   ├── 04_validate_results.py   驗證輸出檔格式
│   └── 05_quick_report.py       在終端機印出摘要（不需網頁也能看結果）
│
├── data/                        資料（不進版控）
│   ├── README.md
│   ├── raw/                     Common Voice 原始下載檔
│   │   ├── archives/            原始 .tar.gz（保留，可重新解壓）
│   │   ├── female/              女聲解壓結果（test.tsv、clips/…）
│   │   └── male/                男聲解壓結果
│   ├── noise/                   噪音源
│   ├── clean/                   50 句 16kHz WAV + manifest.csv
│   └── degraded/                各條件音檔：degraded/<condition>/<utt_id>.wav
│
├── results/                     實驗輸出（每輪一個資料夾）
│   ├── README.md
│   └── round1/
│       ├── results.csv
│       ├── resources.csv
│       ├── meta.json
│       └── checkpoint.json      續跑用
│
├── dashboard/                   結果網頁相關說明
│   ├── README.md
│   ├── data_contract.md         網頁需要的欄位與命名規則
│   └── sections.md              網頁每個區塊的定義與計算方式
│
├── tests/                       單元測試與冒煙測試
│   ├── README.md
│   ├── test_align.py
│   ├── test_wer.py
│   ├── test_textnorm.py
│   ├── test_degrade.py
│   ├── test_manifest.py
│   └── test_smoke_pipeline.py   用 2 句、1 個模型跑完整流程
│
└── docs/                        文件
    ├── README.md
    ├── plan.md                  本計畫規格書
    ├── decisions.md             決策紀錄（為什麼選這些模型、資料集）
    └── experiment_log.md        每次實驗的紀錄與觀察
```

### 4.1 分層原則

| 層 | 職責 | 可以依賴 |
|---|---|---|
| `scripts/` | 解析命令列參數、呼叫 library | `library` |
| `library/runner/` | 組合流程 | 其他所有 library 子模組 |
| `library/{data, degrade, methods, models, textnorm, scoring, monitor, export}` | 單一職責 | 只能依賴 `library/utils/` |
| `library/utils/` | 共用工具 | 無 |

**規則**：子模組之間不互相 import（例如 `scoring` 不能 import `models`），只有 `runner` 負責串接。這樣每個子模組都能被單獨拿去用。

---

## 5. Library 公開 API 規格

所有對外函式都從各子模組的 `__init__.py` 匯出；他人只需 `from library.xxx import ...`。

### 5.1 `library`（高階 API）

給「只想快速用」的人：

```python
def transcribe_file(
    wav_path: str | Path,
    model: str = "whisper-small",
    method: str = "baseline",
    device: str = "cpu",
) -> str:
    """讀取音檔 → 套用前處理方法 → 辨識，回傳文字。"""

def evaluate(ref: str, hyp: str, normalizer: str = "english") -> AlignmentResult:
    """正規化後對齊 ref 與 hyp，回傳 WER 與 S/D/I。"""

def run_experiment(config_path: str | Path) -> Path:
    """依 YAML 設定跑完整實驗矩陣，回傳結果資料夾路徑。"""
```

### 5.2 `library.data`

```python
@dataclass(frozen=True)
class Utterance:
    utt_id: str          # 例：utt001
    wav_path: Path       # clean WAV 路徑
    ref_raw: str         # 原始 reference 文字
    speaker_id: str      # 匿名化 ID（spk01…），不保留 Common Voice client_id
    gender: str          # "female" | "male"
    audio_sec: float

def load_commonvoice_tsv(tsv_path: str | Path, clips_dir: str | Path) -> pd.DataFrame:
    """讀取 Common Voice TSV，回傳含 path、sentence、client_id、gender 的 DataFrame。

    - 必須用 pandas.read_csv(sep="\t", encoding="utf-8") 等 CSV 解析器讀取：
      sentence 欄位含雙引號跳脫（例："...for ""singing"" the..."），不可直接用 split。
    - gender 原始值為 female_feminine / male_masculine，轉為 female / male。
    - client_id 為長雜湊，不輸出；由 build_manifest 轉成匿名 speaker_id。"""

def sample_utterances(
    df: pd.DataFrame,
    n_per_gender: int = 25,
    min_sec: float = 3.0,
    max_sec: float = 8.0,
    min_words: int = 5,
    max_down_votes: int = 0,
    one_per_speaker: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """依規則抽樣；同一 seed 必須得到相同結果。

    TSV 沒有時長欄位，時長需讀音檔取得。為避免讀完上萬個 MP3：
    先依詞數、down_votes、每人一句篩選並以 seed 打亂，
    再依序讀取時長，湊滿 n_per_gender 句就停止。"""

def get_duration(mp3_path: Path) -> float:
    """讀取音檔時長（秒）。"""

def convert_to_wav(src: Path, dst: Path, sr: int = 16000, mono: bool = True) -> float:
    """轉檔並回傳音檔秒數。"""

def build_manifest(samples: pd.DataFrame, out_dir: Path) -> Path:
    """轉檔所有樣本並寫出 manifest.csv。"""

def load_manifest(path: str | Path) -> list[Utterance]: ...
def validate_manifest(path: str | Path) -> list[str]:
    """回傳問題清單（空清單代表通過）：檔案是否存在、取樣率、重複 ID 等。"""
```

### 5.3 `library.degrade`

```python
def add_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """將噪音縮放到指定 SNR 後疊加；噪音長度不足時隨機起點循環補齊。"""

def add_reverb(speech: np.ndarray, sr: int, rt60: float, room_dim: list[float],
               src_mic_distance: float, seed: int) -> np.ndarray: ...

def change_volume(speech: np.ndarray, gain_db: float) -> np.ndarray: ...

def make_noise_only(noise: np.ndarray, sr: int, n_clips: int, clip_sec: float, seed: int) -> list[np.ndarray]: ...

def apply_condition(speech: np.ndarray, sr: int, condition: str, cfg: dict, seed: int) -> np.ndarray:
    """依條件名稱（clean / snr20 / reverb …）套用對應處理。"""

def list_conditions() -> list[str]: ...
```

### 5.4 `library.methods`

```python
class AudioMethod(ABC):
    name: str
    def load(self, device: str = "cpu") -> None: ...       # 載入模型（如 VAD、降噪模型）
    @abstractmethod
    def process(self, wav: np.ndarray, sr: int) -> np.ndarray: ...
    def unload(self) -> None: ...

def get_method(spec: str, config: dict | None = None) -> AudioMethod:
    """spec 可為單一名稱或以 '+' 串接，例如 'vad+denoise' → MethodChain([VAD, Denoise])。
    串接順序即處理順序。"""

def list_methods() -> list[str]: ...
def register_method(name: str, cls: type[AudioMethod]) -> None: ...
```

**特殊規則**：VAD 若判斷整段沒有語音，回傳**長度為 0 的陣列**；runner 收到空音訊時不呼叫模型，直接記錄空字串 hyp。這是幻覺測試的關鍵。

### 5.5 `library.models`

```python
@dataclass
class Transcript:
    text: str                     # 模型原始輸出文字
    language: str | None = None   # 多語模型偵測到的語言
    extra: dict | None = None     # 模型特有資訊（例如 Whisper 的 no_speech_prob）

class ASRModel(ABC):
    name: str
    framework: str
    def __init__(self, config: dict): ...
    @abstractmethod
    def load(self, device: str = "cpu", dtype: str = "fp32", num_threads: int = 4) -> None: ...
    @abstractmethod
    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> Transcript: ...
    def unload(self) -> None: ...         # 釋放記憶體（del 模型 + gc.collect；有 GPU 時另呼叫 torch.cuda.empty_cache）
    def describe(self) -> dict: ...       # 回傳 framework、decoding 等，寫入 meta.json

def get_model(name: str, config: dict | None = None) -> ASRModel: ...
def list_models() -> list[str]: ...
def register_model(name: str, cls: type[ASRModel]) -> None: ...
```

### 5.6 `library.textnorm`

```python
def normalize(text: str, lang: str = "en") -> str:
    """英文：小寫、去標點、縮寫展開、數字統一、英美拼法統一、移除贅詞（uh, um…）。"""

def get_normalizer(name: str = "english") -> Callable[[str], str]: ...
```

### 5.7 `library.scoring`

```python
@dataclass
class AlignmentResult:
    S: int
    D: int
    I: int
    N: int                         # reference 詞數
    ops: list[tuple[str, str, str]]  # (op, ref_word, hyp_word)，op ∈ {"eq","sub","del","ins"}
    @property
    def wer(self) -> float: ...     # N = 0 時回傳 None

def align(ref: str, hyp: str) -> AlignmentResult:
    """輸入應為正規化後的文字。"""

def corpus_wer(results: Iterable[AlignmentResult]) -> float:
    """整體 WER = ΣS+D+I ÷ ΣN（不是逐句 WER 的平均）。"""

def paired_bootstrap(base_errors: list[int], method_errors: list[int], n_words: list[int],
                     n_boot: int = 1000, seed: int = 12345) -> tuple[float, float]:
    """回傳 ΔWER（百分點）的 95% 信賴區間。"""

def compare_per_utterance(base_errors: list[int], method_errors: list[int]) -> dict:
    """回傳 {'improved': n, 'worse': n, 'same': n}。"""
```

### 5.8 `library.monitor`

```python
class ResourceMonitor:
    """Context manager。進入時記錄基準，離開時記錄峰值。
    RAM（主要指標）：psutil 背景執行緒輪詢行程 RSS（每 50ms）。
    GPU（選用）：偵測到 CUDA 時才量；PyTorch 用 torch.cuda.max_memory_allocated，
         其他框架用 pynvml 輪詢。無 GPU 時 peak_gpu_mb = None。"""
    peak_gpu_mb: float | None
    peak_ram_mb: float
    elapsed_ms: float
    def __enter__(self) -> "ResourceMonitor": ...
    def __exit__(self, *exc) -> None: ...

class Timer:
    """with Timer() as t: ... ; t.ms"""
```

### 5.9 `library.runner`

```python
@dataclass
class ExperimentConfig:
    name: str
    manifest: Path
    degraded_dir: Path
    output_dir: Path
    models: list[str]
    methods: list[str]
    conditions: list[str]
    device: str = "cpu"           # Jetson 上改為 "cuda"
    dtype: str = "fp32"           # Jetson 部署預計為 q4，另行複驗（見 11.5）
    num_threads: int = 4          # CPU 執行緒數（i3-10110U 為 2 核 4 緒）
    seed: int = 42
    save_processed: bool = False

def load_config(path: str | Path) -> ExperimentConfig: ...
def iter_matrix(cfg: ExperimentConfig) -> Iterator[tuple[str, str, str, Utterance]]: ...
def run_experiment(cfg: ExperimentConfig, resume: bool = True) -> Path: ...
```

### 5.10 `library.export`

```python
@dataclass
class ResultRow: ...      # 對應 results.csv 一列（見 6.2）
@dataclass
class ResourceRow: ...    # 對應 resources.csv 一列（見 6.3）

class ResultWriter:
    """逐列附加寫入；每寫一列就 flush，確保中斷時資料不遺失。"""
    def __init__(self, path: Path): ...
    def write(self, row: ResultRow) -> None: ...

def write_resources(rows: list[ResourceRow], path: Path) -> None: ...
def write_meta(cfg: ExperimentConfig, models: list[ASRModel], path: Path) -> None: ...
def validate_results(results_dir: Path) -> list[str]: ...
```

---

## 6. 資料格式規格

### 6.1 `data/clean/manifest.csv`

| 欄位 | 型別 | 說明 |
|---|---|---|
| utt_id | str | `utt001`…`utt050`；純噪音為 `noise01`…`noise10` |
| wav_path | str | 相對於專案根目錄的路徑 |
| ref_raw | str | Common Voice 原始句子 |
| speaker_id | str | 匿名 ID（`spk01`…），純噪音留空 |
| gender | str | `female` / `male`（由 female_feminine / male_masculine 轉換），純噪音留空 |
| age | str | Common Voice 年齡層（例：thirties），可留空 |
| audio_sec | float | 秒數 |
| source_file | str | 原始 MP3 檔名（追溯用，不對外公開） |

### 6.2 `results.csv`（每列一個「模型 × 條件 × 方法 × 句子」）

| 欄位 | 型別 | 說明 |
|---|---|---|
| model | str | 見 6.5 命名 |
| condition | str | 見 6.5 命名 |
| method | str | 見 6.5 命名 |
| utt_id | str | |
| speaker_id | str | |
| gender | str | |
| audio_sec | float | 原始音檔秒數（計算 RTF 用） |
| ref_raw | str | 純噪音為空 |
| hyp_raw | str | 模型原始輸出 |
| ref_norm | str | 正規化後 |
| hyp_norm | str | 正規化後 |
| wer | float | 0–1；純噪音為空 |
| S / D / I / N | int | 純噪音：S=D=0、I=hyp 詞數、N=0 |
| latency_ms | float | 方法處理 + 模型辨識的總時間 |
| method_ms | float | 僅方法處理時間 |
| detected_lang | str | 多語模型偵測語言，其他留空 |

### 6.3 `resources.csv`（每列一個「模型 × 方法」）

| 欄位 | 型別 | 說明 |
|---|---|---|
| model / method | str | |
| peak_gpu_mb | float | 該組合執行期間 GPU 峰值；CPU only 時留空 |
| peak_ram_mb | float | 該組合執行期間 RAM 峰值（本輪主要記憶體指標） |
| model_size_mb | float | 模型權重檔大小 |
| load_time_s | float | 模型載入時間 |
| gpu_measure | str | `none`（CPU only）、`torch` 或 `pynvml`（不同測法不可直接比較） |

### 6.4 `meta.json`

```json
{
  "round": "round1",
  "date": "2026-11-xx",
  "git_commit": "abc1234",
  "device": "cpu",
  "cpu": "Intel Core i3-10110U (2C/4T)",
  "ram_gb": 15.8,
  "gpu": "none",
  "num_threads": 4,
  "env": "Windows 11（原生）",
  "python": "3.11.x",
  "precision": "fp32",
  "dataset": "Common Voice Scripted Speech 26.0 US English F+M test, 25+25, seed 42, + 10 noise-only",
  "normalizer": "whisper EnglishTextNormalizer + filler removal",
  "nano_budget_mb": 8192,
  "packages": { "torch": "x.y", "transformers": "x.y" },
  "models": {
    "whisper-small": { "hf_id": "openai/whisper-small", "framework": "transformers", "decoding": "beam 5, temperature 0" }
  },
  "methods": { "vad": { "threshold": 0.5, "min_silence_ms": 300, "pad_ms": 200 } },
  "conditions": { "snr10": { "snr_db": 10, "noise_source": "DEMAND (16k)" } }
}
```

### 6.5 命名規則（網頁依此自動分組，不可隨意更改）

| 欄位 | 允許值 |
|---|---|
| model | `granite-470m`、`whisper-small`、`qwen3-asr-0.6b`（`parakeet-tdt-v2` 保留名稱，暫不使用） |
| condition | `clean`、`snr20`、`snr10`、`snr05`、`snr00`、`reverb`、`lowvol`、`noise_only` |
| method | `baseline`、`vad`、`denoise`、`loudnorm`；組合以 `+` 連接，如 `vad+denoise` |

---

## 7. 設定檔規格

### 7.1 `configs/round1.yaml`

```yaml
name: round1
seed: 42
device: cpu          # Jetson 上改為 cuda
dtype: fp32          # Jetson 部署預計為 q4，以部署版本另行複驗（11.5）
num_threads: 4       # M0 實測：2 緒時 Whisper 慢 65%，維持 4

paths:
  raw:                                 # 依 tar -tzf 結果確認實際路徑
    female: data/raw/female
    male: data/raw/male
  manifest: data/clean/manifest.csv
  degraded_dir: data/degraded
  output_dir: results/round1

sampling:
  tsv_name: test.tsv
  n_per_gender: 25
  min_sec: 3.0
  max_sec: 8.0
  min_words: 5
  max_down_votes: 0
  one_per_speaker: true

models:
  - granite-470m
  # - parakeet-tdt-v2   # 暫緩：需 Linux 環境（NeMo）
  - whisper-small
  - qwen3-asr-0.6b

methods:
  - baseline
  - vad
  - denoise
  - vad+denoise

conditions: [clean, snr20, snr10, snr05, snr00, reverb, lowvol, noise_only]

normalizer: english
save_processed: false
```

### 7.2 `configs/conditions.yaml`

```yaml
noise_source: data/noise/source        # DEMAND 16kHz 數個環境，或機器人本體錄音
conditions:
  clean:   {}
  snr20:   { type: noise, snr_db: 20 }
  snr10:   { type: noise, snr_db: 10 }
  snr05:   { type: noise, snr_db: 5 }
  snr00:   { type: noise, snr_db: 0 }
  reverb:  { type: reverb, rt60: 0.6, room_dim: [5.0, 4.0, 3.0], src_mic_distance: 2.5 }
  lowvol:  { type: volume, gain_db: -20 }
  noise_only: { type: noise_only, n_clips: 10, clip_sec: 5.0 }
```

### 7.3 `configs/models/whisper-small.yaml`（範例）

```yaml
name: whisper-small
hf_id: openai/whisper-small
framework: transformers        # 或 faster-whisper
language: en                   # 本輪固定英文；設 null 則自動偵測
decoding:
  beam_size: 5
  temperature: 0.0
  condition_on_previous_text: false
```

### 7.4 `configs/methods/vad.yaml`（範例）

```yaml
name: vad
impl: silero
threshold: 0.5
min_speech_ms: 250
min_silence_ms: 300
pad_ms: 200          # 語音段前後保留，避免吃掉句首句尾
```

---

## 8. 實驗設計

### 8.1 模型

| 名稱 | Hugging Face ID | 架構 | 框架 | 解碼（固定） |
|---|---|---|---|---|
| granite-470m | ibm-granite/granite-speech-5.0-470m-turboctc | Conformer + CTC | transformers | greedy |
| ~~parakeet-tdt-v2~~（暫緩） | nvidia/parakeet-tdt-0.6b-v2 | FastConformer + TDT | NeMo | greedy |
| whisper-small | openai/whisper-small | Encoder-decoder | transformers / faster-whisper | beam 5, temp 0 |
| qwen3-asr-0.6b | Qwen/Qwen3-ASR-0.6B-hf | Audio-LLM | transformers（≥ 5.13 原生支援） | greedy, language=English |

> 實際的 HF ID 與套件版本在 M0 階段確認後寫入設定檔（已確認，理由見 `docs/decisions.md`）：
> Granite 需 transformers ≥ 5.16；qwen-asr 套件鎖 transformers 4.57，與 Granite 衝突，故改用 transformers 原生版；
> Whisper 由 large-v3-turbo 改為 small（CPU 速度）。

### 8.2 劣化條件

| 條件 | 做法 | 目的 |
|---|---|---|
| clean | 原始音檔 | 基準 |
| snr20 / 10 / 05 / 00 | 疊加噪音至指定 SNR | 抗噪曲線 |
| reverb | RT60 0.6s、距離 2.5m 模擬 | 遠場情境 |
| lowvol | 音量 −20 dB | 小聲 / 距離遠 |
| noise_only | 10 段純噪音，每段 5 秒 | 幻覺率 |

**SNR 計算**：`noise_scaled = noise × sqrt(P_speech / (P_noise × 10^(snr/10)))`，P 為整段平均功率。噪音片段起點以 seed 決定，確保每次產生相同結果。純噪音片段需取自**和疊加用不同的區段**。

**噪音源**：優先使用機器人本體（風扇、馬達）錄音；暫無時使用 DEMAND 16kHz 版，挑 3–4 個室內環境（如客廳、廚房、辦公室）。不用 MUSAN：完整檔約 11GB，D 槽空間不足。

### 8.3 方法

| 方法 | 實作 | 主要參數 | 預期效果 |
|---|---|---|---|
| baseline | 不處理 | — | 基準 |
| vad | Silero VAD | threshold 0.5、pad 200ms | 減少插入錯誤與幻覺，Whisper 最明顯 |
| denoise | noisereduce（原規劃 DeepFilterNet；M0 實測 0.5.6 與新版 torchaudio 不相容，改用備案） | 預設 | 低 SNR 改善；高 SNR 可能變差 |
| loudnorm | pyloudnorm 正規化到 −23 LUFS | 目標響度 | 改善 lowvol 條件 |
| vad+denoise | 先 VAD 再降噪 | — | 兩者疊加效果 |

### 8.4 各實驗的假設與判讀

| 實驗 | 假設 | 看哪些結果 |
|---|---|---|
| Exp0 正規化 | 未正規化 WER 有相當比例是假錯誤 | 正規化影響表 |
| Exp1 VAD + 音量 | VAD 降低插入與幻覺；loudnorm 改善 lowvol | I 比例、幻覺率、lowvol 欄 |
| Exp2 降噪 × SNR | 低 SNR 有效、高 SNR 可能變差 | 抗噪曲線、成本效益表各條件 |

### 8.5 執行規模

3 模型 × 8 條件 × 4 方法 = 96 組；每組 50 句（noise_only 為 10 段），合計約 4,320 次推論。（Parakeet 加入後為 128 組、5,760 次）

**CPU 執行時間評估**

開發機為 i3-10110U，Whisper 與 Qwen3-ASR 這類自回歸模型在 CPU 上每句可能需數秒，完整矩陣預估需要數小時，建議**夜間執行**並依賴續跑機制。M0 階段必須先實測：

1. 每個模型各跑 5 句 clean 音檔，記錄平均每句秒數
2. 預估總時間 = 平均每句秒數 × 每模型 1,440 次推論，三個模型加總
3. 依結果決定：

| 預估總時間 | 做法 |
|---|---|
| ≤ 10 小時 | 照原矩陣，夜間執行 |
| 10–24 小時 | 條件先縮為 clean、snr10、snr00、reverb、lowvol、noise_only；Whisper 改用 faster-whisper int8 |
| > 24 小時 | 再減為每性別 15 句（共 30 句），或分模型分天執行 |

> 若改用 int8 或縮減規模，必須寫入 `meta.json`，並在報告中註明。

**M0 實測結果（2026-10-30，fp32、4 緒、每模型 5 句）**

| 模型 | 平均每句秒數 | RTF | 每模型 1,440 次 |
|---|---|---|---|
| granite-470m | 1.14 | 0.22 | 0.5 小時 |
| whisper-large-v3-turbo（原規劃） | 24.07 | 4.71 | 9.6 小時 |
| whisper-large-v3-turbo，faster-whisper int8 | 28.23 | 5.52 | 11.3 小時（反而較慢，不採用） |
| **whisper-small（採用）** | 11.57 | 2.26 | 4.6 小時 |
| qwen3-asr-0.6b | 5.20 | 1.02 | 2.1 小時 |

決定：保留完整 8 條件 × 4 方法 × 50 句，Whisper 改 small，合計約 7.2 小時（未含方法處理時間），夜間執行。
2 緒時 Whisper 慢 65%，維持 `num_threads: 4`。

---

## 9. 指標定義

| 指標 | 定義 |
|---|---|
| WER | (S + D + I) ÷ N，以正規化後文字計算 |
| 整體 WER | ΣS+D+I ÷ ΣN（加權，非逐句平均） |
| S / D / I 比例 | 各自 ÷ N，加總等於 WER |
| ΔWER | 方法 WER − 同模型同條件 baseline WER（百分點），負值代表改善 |
| 95% CI | 以句為單位配對 bootstrap 1000 次的 2.5% 與 97.5% 分位數 |
| 結論判定 | CI 上界 < 0 → 顯著改善；CI 下界 > 0 → 顯著變差；其餘 → 無顯著差異 |
| 變好 / 變差 / 不變 | 逐句比較錯誤數 |
| 幻覺率 | noise_only 片段中輸出非空文字的比例 |
| 未正規化 WER | 原始文字以空白切詞直接比對 |
| 假錯誤 | 未正規化 WER − 正規化 WER |
| 峰值記憶體 | 每個「模型 × 方法」執行期間 GPU 與 RAM 峰值 |
| 延遲 | 單句方法處理 + 模型辨識時間（ms） |
| RTF | 處理時間 ÷ 音檔長度 |

---

## 10. 結果網頁規格

### 10.1 定位

互動式結果網頁，讀取 `results.csv`、`resources.csv`、`meta.json` 三個檔案，所有彙總（整體 WER、信賴區間、變好/變差、對齊）都由網頁從逐句資料自動計算。**實驗程式只需輸出逐句原始結果。**

範本（假資料版）已建立，格式確認完成。

### 10.2 頁面結構

| 分頁 | 區塊 | 內容 |
|---|---|---|
| **總覽** | 一句話結論 | 自動找出改善最多的「模型 + 方法」與其 WER 變化 |
| | Baseline 總覽 | 各模型 clean / 全條件 WER、幻覺率、峰值 GPU / RAM、模型大小、載入時間、平均延遲 |
| | 正規化影響 | 未正規化 vs 正規化 WER、假錯誤百分點與占比 |
| | 實驗設定 | meta.json 內容、各模型框架與解碼設定 |
| **方法成本效益** | 成本效益總表 | ΔWER、95% CI、Δ GPU、Δ RAM、Δ 延遲、變好 / 變差 / 不變、結論；可切換條件 |
| | WER 矩陣 | 模型 × 方法 × 條件 WER 熱度表 |
| **抗噪與錯誤類型** | 抗噪曲線 | 各模型小圖（共用縱軸），各方法一條線 |
| | S / D / I 拆解 | 選定模型與條件下各方法的錯誤組成 |
| | 幻覺率 | 模型 × 方法長條圖 |
| **資源** | 峰值記憶體 | GPU + RAM 堆疊長條，標示 Nano 8GB 預算線，超過以紅色顯示 |
| | 平均延遲 | 模型 × 方法長條圖 |
| **逐句分析** | 最差 5 句 | ref / hyp 對齊（替換、刪除、插入以顏色標示） |
| | 逐句瀏覽 | 依模型 / 條件 / 方法篩選，可搜尋 |

### 10.3 Round 1 新增項目

因資料集改為 Common Voice，新增：

- **男女聲 WER 比較**（依 `gender` 欄）
- **RTF**（依 `audio_sec` 欄）
- **CPU only 呈現**：`peak_gpu_mb` 留空時，記憶體圖只顯示 RAM，表格中 GPU 欄顯示「—」；延遲圖加註「開發機 CPU 測量，僅供方法間相對比較」

### 10.4 更新流程

1. 執行 `scripts/04_validate_results.py` 確認三個檔案符合規格
2. 將 `results/round1/` 下三個檔案交給 Claude
3. Claude 替換網頁資料來源並加入新欄位對應的圖表
4. 確認網頁上的一句話結論與 `scripts/05_quick_report.py` 終端機摘要一致

### 10.5 注意事項

- 網頁只放轉寫文字與分數，**不放音檔**（Common Voice 條款）
- `speaker_id` 使用匿名 ID，不使用 Common Voice 的 client_id
- 網頁預設為私人，分享前需確認內容

---

## 11. 環境建置

### 11.1 前置需求（Windows 原生、CPU only）

| 項目 | 安裝方式 | 確認指令（PowerShell） |
|---|---|---|
| uv | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"`，裝完重開 PowerShell | `uv --version` |
| Python 3.11 | 由 uv 安裝，不需另外下載：`uv python install 3.11` | `uv python list` |
| ffmpeg | `winget install Gyan.FFmpeg`，裝完重開 PowerShell | `ffmpeg -version` |
| Git（選用） | `winget install Git.Git` | `git --version` |

> 不需要 NVIDIA 驅動與 CUDA。

### 11.2 建立 uv 專案環境（PowerShell）

```powershell
cd "D:\stt test"

# （選用）C 槽空間不足時，把 uv 與 Hugging Face 快取移到 D 槽
[Environment]::SetEnvironmentVariable("UV_CACHE_DIR", "D:\stt test\.uv_cache", "User")
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\stt test\.hf_cache", "User")
# 設定後重開 PowerShell 才會生效

uv python install 3.11
uv init --python 3.11 --no-readme      # 產生 pyproject.toml（若骨架已有 pyproject.toml 則略過）
uv sync                                # 依 pyproject.toml 建立 .venv 並安裝套件
uv run python scripts\check_env.py     # 之後所有腳本都用 uv run 執行
```

**PyTorch 需指定 CPU 版**（避免下載數 GB 的 CUDA 版）。在 `pyproject.toml` 加入：

```toml
[project]
name = "stt-wer-bench"
requires-python = ">=3.11,<3.12"
dependencies = [
  "numpy", "pandas", "soundfile", "librosa", "pyyaml", "jiwer", "psutil", "tqdm",
  "torch", "torchaudio", "transformers>=5.16",
]

[project.optional-dependencies]
audio  = ["pyroomacoustics", "pyloudnorm", "silero-vad", "deepfilternet", "noisereduce"]
models = ["faster-whisper"]   # M0：移除 qwen-asr（鎖 transformers 4.57，與 Granite 衝突）

[dependency-groups]
dev = ["pytest", "ruff", "mypy"]

[tool.uv.sources]
torch      = { index = "pytorch-cpu" }
torchaudio = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

```powershell
uv sync --extra audio --extra models    # 安裝選用群組
uv add <套件>                           # 新增套件（自動寫入 pyproject.toml 與 uv.lock）
```

- `uv.lock` 要進版控，確保之後（含 Jetson）能裝出相同版本。
- 不需要啟動 `.venv`；`uv run` 會自動使用專案環境。

### 11.3 相依套件分組

| 群組 | 套件 |
|---|---|
| 主要（dependencies） | numpy、pandas、soundfile、librosa、pyyaml、jiwer、psutil、tqdm、torch（CPU 版）、torchaudio、transformers（≥ 5.16，Granite 與 Qwen3-ASR 原生支援）、whisper-normalizer（或 openai-whisper 內建 normalizer） |
| audio | pyroomacoustics、pyloudnorm、silero-vad、noisereduce（實際使用）；deepfilternet 已安裝但 import 失敗，不使用 |
| models | faster-whisper（int8 備案；M0 實測此 CPU 上不會加速） |
| dev | pytest、ruff、mypy |
| 暫不安裝 | nemo_toolkit（Parakeet）、vLLM（不支援 Windows）、KenLM、pynvml（無 NVIDIA 顯卡） |

> 版本號由 `uv.lock` 鎖定，並在 M0 完成時寫入 `meta.json`。

**Windows 注意事項**

- 專案路徑 `D:\stt test` **含空格**：命令列中的路徑一律加引號；程式內一律用 `pathlib.Path`，不要自己拼接字串。
- 台灣 Windows 預設編碼為 cp950：讀寫 CSV / JSON / TSV **一律指定 `encoding="utf-8"`**，否則英文以外的字元會亂碼或報錯。
- 使用多程序（multiprocessing、DataLoader `num_workers>0`）的腳本必須有 `if __name__ == "__main__":` 保護。
- Hugging Face 下載模型時可能出現 symlink 警告，不影響使用；快取位置見 11.2。三個模型 fp32 權重合計約 7–8GB，下載前確認空間。
- CPU 推論時用 `torch.set_num_threads(num_threads)` 固定執行緒數，並在執行期間避免開啟其他吃 CPU 的程式，否則延遲數字會浮動。
- 若 DeepFilterNet 安裝失敗，降噪方法改用 noisereduce，並在 `meta.json` 記錄實際使用的實作。

### 11.4 資料下載與放置

**資料集需手動下載**：Mozilla Data Collective 需登入並同意條款，無法由程式或 Claude Code 自動下載。

| 項目 | 誰來做 | 方式 |
|---|---|---|
| Common Voice 女聲 / 男聲 .tar.gz | **手動**（已完成） | 已下載到 `D:\stt test\` |
| 噪音源（DEMAND 16kHz） | Claude Code 可代勞 | 下載 3–4 個室內環境（或使用機器人錄音） |
| 三個模型權重 | Claude Code 可代勞 | 首次載入時由 Hugging Face 自動下載 |

**放置規則**

1. 原始壓縮檔移到 `data\raw\archives\`，解壓到 `data\raw\female\`、`data\raw\male\`。
2. 兩個壓縮檔合計約 14GB，解壓後再佔一份空間，先確認 D 槽剩餘空間。
3. 解壓幾萬個小檔時，Windows Defender 即時掃描會明顯拖慢速度；可暫時將 `D:\stt test\data` 加入排除清單（選用）。

```powershell
cd "D:\stt test"
New-Item -ItemType Directory -Force data\raw\archives, data\raw\female, data\raw\male

Move-Item ".\1784537601551-cv26-american-english-female.tar.gz" "data\raw\archives\cv26_us_female.tar.gz"
Move-Item ".\1784540315873-cv26-american-english-male.tar.gz"   "data\raw\archives\cv26_us_male.tar.gz"

# 先看內部結構（Windows 10 / 11 內建 tar），確認 TSV 與 clips 的相對路徑
tar -tzf "data\raw\archives\cv26_us_female.tar.gz" | Select-Object -First 30

# 完整解壓
tar -xzf "data\raw\archives\cv26_us_female.tar.gz" -C "data\raw\female"
tar -xzf "data\raw\archives\cv26_us_male.tar.gz"   -C "data\raw\male"
```

**實際採用：只解出 test split（D 槽剩約 18GB，完整解壓會幾乎塞滿）**

壓縮檔內部結構（已確認）：

```
dev.tsv
test.tsv
train.tsv
clips/common_voice_en_xxxxx.mp3
```

執行的指令（女聲、男聲依序處理；第 3 步會掃完整個壓縮檔，期間沒有輸出屬正常）：

```powershell
cd "D:\stt test"
foreach ($g in "female", "male") {
    $arc = "data\raw\archives\cv26_us_$g.tar.gz"
    $dst = "data\raw\$g"
    tar -x -z -q -f $arc -C $dst test.tsv                       # 1. 只解 test.tsv
    $lines = Get-Content "$dst\test.tsv" -Encoding UTF8          # 2. 由 path 欄產生清單
    $i = [array]::IndexOf($lines[0].Split("`t"), "path")
    $list = $lines | Select-Object -Skip 1 | ForEach-Object {
        $p = $_.Split("`t")[$i]; if ($p -like "clips/*") { $p } else { "clips/$p" } }
    [IO.File]::WriteAllText("$PWD\$dst\clip_list.txt", ($list -join "`n") + "`n")   # LF 換行
    tar -x -z -f $arc -C $dst -T "$dst\clip_list.txt"           # 3. 只解清單中的音檔
    "$g：清單 $($list.Count) 筆，實際解出 $((Get-ChildItem "$dst\clips").Count) 個"
}
```

結果（2026-10-30 完成，數量皆與清單一致）：女聲 test 2,888 句、男聲 test 7,485 句。

**test.tsv 欄位（已確認）**

| 欄位 | 用途 |
|---|---|
| client_id | 說話者雜湊 → 轉匿名 speaker_id，不對外輸出 |
| path | 音檔名（不含 `clips/`） |
| sentence | reference 文字（含雙引號跳脫，需用 CSV 解析器讀） |
| up_votes / down_votes | 驗證票數；抽樣只取 down_votes = 0 |
| age | 年齡層（之後可作分析維度） |
| gender | `female_feminine` / `male_masculine` |
| accents | `United States English` |
| sentence_id、sentence_domain、variant、locale、segment | 本輪不使用 |

**沒有時長欄位**：時長需由程式讀音檔取得（見 5.2 `get_duration`）。

### 11.5 Jetson 遷移（之後）

- 在 Jetson 上以相同設定檔、相同 50 句複跑 baseline 與效果最好的方法，**使用預計部署的 q4 版本**（推論引擎依模型而定）
- 比較「方法開/關的改善幅度」是否與 Windows 一致
- Jetson 為 Linux：先跑 `pytest` 確認跨平台沒問題，再複跑實驗；屆時可一併加入 Parakeet
- 實測記憶體（統一記憶體，需看系統總用量）

---

## 12. 里程碑與驗收標準

| 里程碑 | 內容 | 產出 | 驗收標準 |
|---|---|---|---|
| **M0 環境** | uv 環境、CPU 版 PyTorch、套件、三個模型可載入；CPU 速度實測 | `check_env.py` 通過、速度實測表 | 三個模型各辨識 5 句 clean 音檔成功；依 8.5 節估算總時間並決定規模；`uv.lock` 已產生 |
| **M1 資料準備** | 解壓、抽樣、轉檔、manifest | `data/clean/` | 50 句、25+25、50 位不同說話者；全部 16kHz mono；同 seed 重跑結果相同 |
| **M2 劣化產生** | 7 條件 + 純噪音 | `data/degraded/` | 實測 SNR 與目標誤差 < 0.5 dB；抽聽 5 句確認 |
| **M3 評分模組（Exp0）** | 正規化、對齊、統計 | `textnorm/`、`scoring/` + 測試 | 單元測試通過；與 jiwer 計算結果一致 |
| **M4 模型與方法包裝** | 3 個模型（Parakeet 保留空殼）、4 個方法 | `models/`、`methods/` | 冒煙測試（2 句 × 全組合）跑通 |
| **M5 執行器與監測** | runner、checkpoint、monitor、export | `runner/`、`monitor/`、`export/` | 中斷後重跑能續跑；輸出通過 validate |
| **M6 Round 1 執行** | 完整 96 組 | `results/round1/` | 4,320 列完整；validate 通過 |
| **M7 結果網頁與報告** | 網頁更新、實驗紀錄 | 網頁、`docs/experiment_log.md` | 每個 Exp 都有結論與佐證圖表 |

---

### 12.1 分工：人工 vs Claude Code

| 工作 | 誰做 | 說明 |
|---|---|---|
| 安裝 uv、ffmpeg | **人工** | 各一行指令（見 11.1） |
| 下載 Common Voice | **人工** | 需登入並同意條款 |
| 建立資料夾、README、程式骨架 | Claude Code | 依本規格書第 4、5 章與附錄 A |
| 安裝 Python 套件（uv）、下載模型、下載 DEMAND 噪音 | Claude Code | 執行前會請你確認指令 |
| 實作各模組、測試 | Claude Code | 依里程碑逐步進行 |
| 抽聽劣化音檔、看最差句子 | **人工** | 判斷結果是否合理 |
| 更新結果網頁 | Claude（本對話） | 交付 results/round1/ 三個檔案 |

### 12.2 交給 Claude Code 的方式

1. 在 `D:\stt test` 建立 `docs\` 資料夾，把本規格書放到 `docs\plan.md`
2. 在專案根目錄啟動 Claude Code
3. **一次一個里程碑**，每完成一個先自己檢查再進下一個，不要一次要求做完全部

建議指令範例：

```
請閱讀 docs/plan.md。環境是 Windows 原生（PowerShell）、CPU only（無 GPU），
用 uv 管理環境（PyTorch 用 CPU 版，見 11.2），專案根目錄是 D:\stt test，
Parakeet 暫緩（只建空殼）。

這一步只做「專案骨架」，不要開始實作、不要下載任何模型：
1. 依第 4 章建立所有資料夾與檔案。
2. 依附錄 A 寫入每個資料夾的 README.md。
3. 依第 5 章為每個函式與類別寫好型別標註、簽名與 docstring，
   內容先 raise NotImplementedError。
4. 依第 7 章填入 configs/ 下所有 yaml 的內容（parakeet 設 enabled: false）。
5. 依第 6、10 章寫入 dashboard/data_contract.md 與 dashboard/sections.md。
6. tests/ 下每個測試檔先放一個 pytest.skip 的佔位測試。
7. 依 11.2 建立 pyproject.toml（PyTorch 指定 CPU index），建立 .gitignore
   （排除 data/、.venv/、.uv_cache/、.hf_cache/，uv.lock 要保留）。
8. 在根目錄建立 CLAUDE.md，寫入不可違反的規則：6.5 命名規則、4.1 分層原則、
   第 14 章撰寫規範（pathlib、utf-8、uv run、不假設有 GPU）、不可上傳或散佈 data/。

注意：data\raw\ 下已有解壓好的資料（archives、female、male），
不可刪除、移動或覆寫其中任何檔案，只能在 data\ 下新增缺少的資料夾與 README.md。
所有檔案以 UTF-8 儲存。

完成後：
- 執行 uv sync，確認環境建立成功
- 執行 uv run python -c "import library"，確認骨架可以被 import
- 執行 uv run pytest -q，確認測試全部為 skipped 而非錯誤
- 列出建立的所有檔案，並回報上面三個指令的結果
```

```
請實作 M1（資料準備）：library/data/ 與 scripts/01_prepare_data.py。
資料已放在 data/raw/female 與 data/raw/male（結構見 tar -tzf 結果：<貼上>）。
完成後執行並回報抽樣結果，以及 tests/test_manifest.py 的測試結果。
```

> 建議同時在專案根目錄建立 `CLAUDE.md`，寫入不可違反的規則（命名規則 6.5、分層原則 4.1、不寫死參數、不上傳 data/），讓每次對話都遵守。

## 13. 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| Qwen3-ASR 在 Windows 安裝失敗 | 少一個模型 | 改用 SenseVoice-Small（sherpa-onnx 支援 Windows） |
| Parakeet 暫缺 | 少了 Transducer 架構的比較 | 介面保留；有 Linux 環境後補跑同一份設定 |
| DeepFilterNet 在 Windows 安裝失敗 | 降噪方法無法執行 | 改用 noisereduce，於 meta.json 記錄 |
| 路徑含空格、cp950 編碼 | 檔案讀寫錯誤 | 一律 pathlib、一律 utf-8（見第 14 章） |
| Windows 與 Jetson 環境差異 | 結果搬過去不一致 | 程式維持跨平台；Jetson 上先跑測試再抽樣複驗 |
| 資料污染使 baseline WER 過低 | 方法看不出效果 | 依賴劣化條件拉開差距；報告註明 |
| 50 句統計力不足 | 改善是否真實不確定 | 配對 bootstrap、逐句變好/變差；必要時擴增到 100 句 |
| 不同框架記憶體測法不同 | 記憶體無法公平比較 | `gpu_measure` 欄位標示；同模型內比較方法差異 |
| 降噪模型本身佔記憶體 | 影響模型峰值記憶體 | 方法與模型分開量測並分別記錄 |
| CPU 推論太慢 | 完整矩陣跑不完 | M0 實測速度，依 8.5 節縮減條件、改 int8 或分天執行；續跑機制 |
| fp32（開發）與 q4（Jetson 部署）精度不同 | 量化模型對噪音的敏感度可能不同，方法改善幅度不一定能直接搬過去 | 看方法間相對改善；Jetson 上以 q4 部署版本複驗 baseline 與最佳方法 |
| D 槽空間不足（剩約 18GB） | 模型、噪音、劣化音檔放不下 | 不用 MUSAN；快取集中管理；劣化音檔只存 50 句 × 8 條件（約數百 MB） |
| Whisper temperature fallback 不固定 | 結果不可重現 | 固定 temperature 0；穩定性抽查（同輸入跑兩次） |
| 長時間執行中斷 | 重跑浪費時間 | checkpoint 續跑、逐列寫入 |

---

## 14. 程式撰寫規範

- Python 3.11、全面使用型別標註
- 每個公開函式要有 docstring：用途、參數、回傳值、例外
- 路徑一律透過 `library/utils/paths.py` 並使用 `pathlib.Path`，不寫死絕對路徑、不手動拼接 `\` 或 `/`
- 讀寫文字檔一律指定 `encoding="utf-8"`（Windows 預設 cp950）
- 程式中不呼叫特定 shell 指令（如 `ls`、`cp`）；需要時用 Python 標準函式庫（`shutil`、`tarfile`），確保 Windows 與 Jetson 都能執行
- 所有 `scripts/` 都要有 `if __name__ == "__main__":` 保護
- 一律以 `uv run` 執行；新增套件用 `uv add`，不直接 `pip install`
- 裝置一律由設定檔 `device` 決定，程式不可假設有 GPU（`cuda` 不可用時自動退回 `cpu` 並記錄警告）
- 參數一律來自設定檔；程式中只放預設值
- 亂數一律透過 `library/utils/seed.py` 並記錄 seed
- 新增模型 / 方法：繼承 base 類別 → 在 registry 註冊 → 新增設定檔 → 更新該資料夾 README
- 每個子模組至少一個單元測試；`test_smoke_pipeline.py` 必須在 1 分鐘內跑完
- 以 ruff 格式化與檢查
- Commit 訊息格式：`[模組] 動作：說明`，例如 `[scoring] add：配對 bootstrap`

---

## 附錄 A：各資料夾 README 內容

以下為每個資料夾 `README.md` 的內容。原則：**說明這個資料夾做什麼、每個檔案做什麼、怎麼呼叫、怎麼擴充**。

---

### A.1 `README.md`（專案根目錄）

```markdown
# stt test（STT WER 測試專案）

找出 STT 錯誤來源並驗證降低 WER 的方法，同時記錄記憶體與延遲代價。

## 快速開始
1. 建環境：見 docs/plan.md 第 11 章（uv sync --extra audio --extra models）
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
```

### A.2 `configs/README.md`

```markdown
# configs/

所有實驗參數集中在這裡，程式不寫死任何參數。

| 檔案 | 用途 |
|---|---|
| round1.yaml | 第一輪總設定：要跑哪些模型、方法、條件，輸出到哪 |
| conditions.yaml | 劣化條件定義（SNR、殘響、音量、純噪音） |
| models/*.yaml | 每個模型的 HF ID、框架、解碼參數 |
| methods/*.yaml | 每個前處理方法的參數 |

## 新增一輪實驗
複製 round1.yaml 為 round2.yaml，修改 name 與 output_dir，其餘依需要調整。
```

### A.3 `library/README.md`

```markdown
# library/

核心函式庫。每個子模組只負責一件事，彼此不互相依賴，只有 runner/ 負責串接。

| 子模組 | 職責 | 主要入口 |
|---|---|---|
| data/ | 讀資料集、抽樣、轉檔、manifest | sample_utterances, build_manifest, load_manifest |
| degrade/ | 產生劣化音訊 | apply_condition, add_noise |
| methods/ | 音訊前處理方法 | get_method("vad+denoise") |
| models/ | STT 模型統一介面 | get_model("whisper-small") |
| textnorm/ | 文字正規化 | normalize |
| scoring/ | 對齊、WER、統計 | align, corpus_wer, paired_bootstrap |
| monitor/ | 記憶體、延遲量測 | ResourceMonitor, Timer |
| runner/ | 實驗矩陣執行 | run_experiment |
| export/ | 輸出結果檔 | ResultWriter, validate_results |
| utils/ | 路徑、log、seed | — |

## 高階 API（__init__.py）
- transcribe_file(wav_path, model, method) → str
- evaluate(ref, hyp) → AlignmentResult
- run_experiment(config_path) → Path
```

### A.4 `library/data/README.md`

```markdown
# library/data/

資料集讀取、抽樣、轉檔與 manifest 管理。

| 檔案 | 做什麼 |
|---|---|
| types.py | Utterance 資料結構（utt_id、wav_path、ref_raw、speaker_id、gender、audio_sec） |
| commonvoice.py | load_commonvoice_tsv()：讀 Common Voice TSV，整理成 DataFrame |
| sampler.py | sample_utterances()：依性別、時長、詞數、每人一句抽樣，固定 seed |
| audio_io.py | read_audio()、write_audio()、convert_to_wav()：統一轉 16kHz mono |
| manifest.py | build_manifest()、load_manifest()、validate_manifest() |

## 範例
    df = load_commonvoice_tsv("data/raw/female/test.tsv", "data/raw/female/clips")
    samples = sample_utterances(df, n_per_gender=25, seed=42)
    build_manifest(samples, Path("data/clean"))

## 注意
- speaker_id 一律匿名化（spk01…），不輸出 Common Voice client_id
- 新增資料集：新增一個讀取檔（例如 mslt.py），輸出相同欄位的 DataFrame 即可
```

### A.5 `library/degrade/README.md`

```markdown
# library/degrade/

產生實驗用的劣化音訊。結果預先存檔，確保每個模型聽到相同音訊。

| 檔案 | 做什麼 |
|---|---|
| noise.py | add_noise()：把噪音縮放到指定 SNR 後疊加 |
| reverb.py | add_reverb()：用 pyroomacoustics 模擬房間殘響 |
| volume.py | change_volume()：音量增減（dB） |
| noise_only.py | make_noise_only()：切出純噪音片段（幻覺測試用） |
| registry.py | 條件名稱對應處理函式；apply_condition()、list_conditions() |

## 新增條件
1. 在 conditions.yaml 新增條件與參數
2. 若是新類型，在本資料夾新增處理檔並於 registry.py 註冊
```

### A.6 `library/methods/README.md`

```markdown
# library/methods/

音訊前處理方法（實驗變因）。所有方法都繼承 AudioMethod，介面為 process(wav, sr) → wav。

| 檔案 | 做什麼 |
|---|---|
| base.py | AudioMethod 抽象類別：load() / process() / unload() |
| registry.py | get_method()、list_methods()、register_method() |
| chain.py | MethodChain：依序串接，支援 "vad+denoise" |
| identity.py | baseline：原樣回傳 |
| vad.py | Silero VAD：保留語音段；完全無語音時回傳空陣列 |
| denoise.py | 降噪（noisereduce；原規劃 DeepFilterNet） |
| loudnorm.py | 響度正規化到目標 LUFS |

## 範例
    m = get_method("vad+denoise"); m.load("cpu")
    out = m.process(wav, 16000)

## 新增方法
1. 新增 xxx.py，繼承 AudioMethod 實作 process()
2. 在 registry.py 註冊名稱
3. 新增 configs/methods/xxx.yaml
4. 更新本 README
```

### A.7 `library/models/README.md`

```markdown
# library/models/

STT 模型包裝。所有模型繼承 ASRModel，介面為 transcribe(wav, sr) → Transcript。

| 檔案 | 模型 | 架構 |
|---|---|---|
| base.py | ASRModel 抽象類別、Transcript 資料結構 | — |
| registry.py | get_model()、list_models()、register_model() | — |
| granite.py | Granite Speech 5.0 470M TurboCTC | CTC |
| parakeet.py | Parakeet TDT 0.6B v2（暫緩，需 Linux／NeMo，目前只有空殼） | Transducer |
| whisper.py | Whisper small | Encoder-decoder |
| qwen3_asr.py | Qwen3-ASR-0.6B | Audio-LLM |

## 範例
    model = get_model("granite-470m"); model.load("cpu", "fp32")
    print(model.transcribe(wav, 16000).text)
    model.unload()

## 新增模型
1. 新增 xxx.py，繼承 ASRModel，實作 load() 與 transcribe()
2. describe() 回傳框架與解碼設定（寫入 meta.json）
3. 在 registry.py 註冊；新增 configs/models/xxx.yaml；更新本 README

## 注意
- transcribe() 不做正規化，回傳模型原始輸出
- unload() 必須釋放記憶體，否則下一個模型的峰值會被污染
```

### A.8 `library/textnorm/README.md`

```markdown
# library/textnorm/

文字正規化。ref 與 hyp 必須使用同一個 normalizer。

| 檔案 | 做什麼 |
|---|---|
| english.py | 小寫、去標點、縮寫展開、數字統一、英美拼法統一、移除贅詞 |
| registry.py | get_normalizer(name)；預留中文（簡轉繁、全半形、數字）版本 |

## 範例
    normalize("I don't know, uh, twenty-five.")  # → "i do not know 25"
```

### A.9 `library/scoring/README.md`

```markdown
# library/scoring/

對齊與評分。輸入必須是正規化後的文字。

| 檔案 | 做什麼 |
|---|---|
| align.py | align()：Levenshtein 對齊，回傳 S/D/I/N 與對齊序列 |
| wer.py | 逐句 WER、corpus_wer()（ΣS+D+I ÷ ΣN） |
| stats.py | paired_bootstrap()：ΔWER 95% CI；compare_per_utterance()：變好/變差/不變 |

## 範例
    r = align("i love you", "i love u")   # S=1, D=0, I=0, N=3, wer=0.333
```

### A.10 `library/monitor/README.md`

```markdown
# library/monitor/

資源與時間量測。

| 檔案 | 做什麼 |
|---|---|
| memory.py | ResourceMonitor：峰值 RAM（psutil 輪詢）；偵測到 GPU 時另量 GPU |
| timer.py | Timer：毫秒計時 |

## 範例
    with ResourceMonitor() as mon:
        model.transcribe(wav)
    print(mon.peak_gpu_mb, mon.peak_ram_mb, mon.elapsed_ms)

## 注意
- 本輪為 CPU only：peak_gpu_mb 為 None，只看 RAM；fp32 權重約為 Jetson 部署 q4 的 8 倍，需在 Jetson 實測
- 不同量測方式（torch / pynvml）的數字不可直接比較
```

### A.11 `library/runner/README.md`

```markdown
# library/runner/

串接所有模組，執行「模型 × 方法 × 條件 × 句子」實驗矩陣。

| 檔案 | 做什麼 |
|---|---|
| config.py | load_config()：讀 YAML 並驗證欄位 |
| matrix.py | iter_matrix()：依「模型 → 方法 → 條件 → 句子」順序展開 |
| checkpoint.py | 記錄已完成組合，重跑時跳過 |
| runner.py | run_experiment()：主流程，逐列寫入結果 |

## 流程
載入模型 → 每個方法重設資源統計 → 每個條件讀取劣化音檔 → 方法處理 → 模型辨識
→ 正規化 → 對齊 → 寫入一列 → 該模型結束後 unload
```

### A.12 `library/export/README.md`

```markdown
# library/export/

輸出結果檔，格式需符合 dashboard/data_contract.md。

| 檔案 | 做什麼 |
|---|---|
| schema.py | ResultRow、ResourceRow 欄位定義 |
| writer.py | ResultWriter（逐列寫入並 flush）、write_resources() |
| meta.py | write_meta()：收集 CPU / GPU、套件版本、git commit、設定 |
| validate.py | validate_results()：檢查欄位、命名、列數、數值範圍 |
```

### A.13 `library/utils/README.md`

```markdown
# library/utils/

| 檔案 | 做什麼 |
|---|---|
| paths.py | 專案根目錄與 data、results 等路徑集中管理 |
| logging.py | 統一 log 格式（時間、模組、等級） |
| seed.py | set_seed()：同時固定 random、numpy、torch |
```

### A.14 `scripts/README.md`

```markdown
# scripts/

命令列入口，只解析參數並呼叫 library，不放邏輯。依編號順序執行。

| 檔案 | 做什麼 | 主要參數 |
|---|---|---|
| check_env.py | 檢查 Python、套件版本、可用裝置、三個模型能否載入，並實測每句秒數 | --models, --bench 5 |
| 01_prepare_data.py | 抽樣 50 句、轉檔、產生 manifest | --config |
| 02_make_degraded.py | 產生所有劣化條件音檔 | --config |
| 03_run_matrix.py | 執行實驗矩陣（可續跑） | --config, --no-resume, --save-processed |
| 04_validate_results.py | 驗證三個輸出檔 | results_dir |
| 05_quick_report.py | 終端機印出各模型 × 方法整體 WER 摘要 | results_dir |
```

### A.15 `data/README.md`

```markdown
# data/

不進版控。Common Voice 條款禁止重新散佈，請勿上傳到公開位置。

| 資料夾 | 內容 |
|---|---|
| raw/archives/ | 手動下載的原始 .tar.gz（保留） |
| raw/female/、raw/male/ | 只解出的 test split：test.tsv、clip_list.txt、clips/ |
| noise/ | 噪音源（機器人錄音或 DEMAND 16kHz） |
| clean/ | 50 句 16kHz mono WAV + manifest.csv |
| degraded/ | degraded/<condition>/<utt_id>.wav |
```

### A.16 `results/README.md`

```markdown
# results/

每輪實驗一個資料夾，例如 round1/。

| 檔案 | 內容 |
|---|---|
| results.csv | 逐句結果（每列一個 模型×條件×方法×句子） |
| resources.csv | 每個 模型×方法 的峰值記憶體、模型大小、載入時間 |
| meta.json | 實驗環境與設定 |
| checkpoint.json | 續跑用，不需交給網頁 |

欄位規格見 dashboard/data_contract.md。
```

### A.17 `dashboard/README.md`

```markdown
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
```

### A.18 `tests/README.md`

```markdown
# tests/

    pytest -q

| 檔案 | 測什麼 |
|---|---|
| test_align.py | S/D/I 計算、空字串、全錯等邊界情況 |
| test_wer.py | corpus_wer 為加權而非平均；與 jiwer 一致 |
| test_textnorm.py | 標點、縮寫、數字、贅詞 |
| test_degrade.py | 實測 SNR 與目標誤差、同 seed 結果相同 |
| test_manifest.py | 欄位、檔案存在、取樣率 |
| test_smoke_pipeline.py | 2 句 × 1 模型跑完整流程（1 分鐘內） |
```

### A.19 `docs/README.md`

```markdown
# docs/

| 檔案 | 內容 |
|---|---|
| plan.md | 計畫規格書 |
| decisions.md | 重要決策與理由（模型、資料集、指標的選擇） |
| experiment_log.md | 每次實驗：日期、設定、觀察、結論、下一步 |
```

---

## 附錄 B：呼叫範例

### B.1 辨識單一音檔

```python
from library import transcribe_file

text = transcribe_file("sample.wav", model="whisper-small", method="vad")
print(text)
```

### B.2 計算單句 WER

```python
from library import evaluate

r = evaluate("I love you.", "i love u")
print(r.wer, r.S, r.D, r.I)    # 0.333 1 0 0
```

### B.3 自行組合流程

```python
from library.data import load_manifest, read_audio
from library.methods import get_method
from library.models import get_model
from library.textnorm import normalize
from library.scoring import align
from library.monitor import ResourceMonitor

utts = load_manifest("data/clean/manifest.csv")
method = get_method("vad+denoise"); method.load("cpu")
model = get_model("granite-470m"); model.load("cpu", "fp32")

with ResourceMonitor() as mon:
    for u in utts[:5]:
        wav, sr = read_audio(u.wav_path)
        hyp = model.transcribe(method.process(wav, sr), sr).text
        r = align(normalize(u.ref_raw), normalize(hyp))
        print(u.utt_id, r.wer)

print(mon.peak_gpu_mb, mon.peak_ram_mb)
model.unload(); method.unload()
```

### B.4 跑完整實驗

```python
from library import run_experiment

out_dir = run_experiment("configs/round1.yaml")
```
