"""喚醒詞偵測：sherpa-onnx KeywordSpotter 的串流包裝（套件於 load() 內 import）。"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from library.utils.paths import resolve_path

_TAIL_SEC = 0.66  # 音檔結尾補靜音，讓最後一個喚醒詞的判斷能完成（sherpa-onnx 範例的做法）


@dataclass(frozen=True)
class Detection:
    """一次喚醒。

    屬性:
        keyword: 喚醒詞標籤（keywords 的 label，例：小咪小咪）。
        t_sec: 偵測到的時間點（相對於串流開頭的秒數，以送入的音訊計，誤差為一個 chunk）。
    """

    keyword: str
    t_sec: float


def keywords_text(keywords: list[dict]) -> str:
    """把設定檔的 keywords 轉成 sherpa-onnx keywords 檔內容（每行 `tokens @label`）。

    參數:
        keywords: [{"tokens": "x iǎo m ī x iǎo m ī", "label": "小咪小咪"}, ...]；
            tokens 為建模單元（中文：聲母 + 帶聲調韻母），以空白分隔。

    回傳:
        keywords 檔內容（結尾含換行）。

    例外:
        ValueError: keywords 為空、缺少 tokens / label，或 label 重複。
    """
    if not keywords:
        raise ValueError("keywords 不可為空")
    lines, labels = [], set()
    for kw in keywords:
        tokens, label = str(kw.get("tokens", "")).split(), str(kw.get("label", "")).strip()
        if not tokens or not label:
            raise ValueError(f"keyword 需要 tokens 與 label：{kw}")
        if label in labels:
            raise ValueError(f"keyword label 重複：{label}")
        labels.add(label)
        lines.append(" ".join(tokens) + " @" + label)
    return "\n".join(lines) + "\n"


def missing_tokens(keywords: list[dict], tokens_file: str | Path) -> list[str]:
    """回傳 keywords 用到、但模型 tokens.txt 沒有的建模單元（有缺時模型無法偵測該詞）。

    參數:
        keywords: 同 keywords_text()。
        tokens_file: 模型的 tokens.txt。

    回傳:
        缺少的 token（依出現順序、不重複）。

    例外:
        FileNotFoundError: tokens_file 不存在。
    """
    path = Path(tokens_file)
    if not path.is_file():
        raise FileNotFoundError(f"tokens.txt 不存在：{path}")
    have = {line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    used = [t for kw in keywords for t in str(kw.get("tokens", "")).split()]
    return [t for t in dict.fromkeys(used) if t not in have]


class KeywordStream:
    """一條連續的音訊串流（例如麥克風）；由 KeywordSpotter.stream() 建立。"""

    def __init__(self, spotter: object, stream: object, sr: int) -> None:
        self._spotter, self._stream, self._sr = spotter, stream, sr
        self._n = 0  # 已送入的樣本數

    def accept(self, chunk: np.ndarray) -> list[Detection]:
        """送入一段音訊，回傳這段期間偵測到的喚醒（偵測後自動重設，可連續喚醒）。

        參數:
            chunk: float32 一維陣列（取樣率與建立串流時相同）。

        回傳:
            Detection 列表（通常為空）。
        """
        kws, s = self._spotter, self._stream
        s.accept_waveform(self._sr, np.ascontiguousarray(chunk, dtype=np.float32))
        self._n += len(chunk)
        hits = []
        while kws.is_ready(s):
            kws.decode_stream(s)
            label = kws.get_result(s)
            if label:
                hits.append(Detection(label, self._n / self._sr))
                kws.reset_stream(s)
        return hits


class KeywordSpotter:
    """sherpa-onnx 喚醒詞偵測。

    設定（configs/round4.yaml 的 kws.models.<名稱> 加上 kws 共用欄位）:
        model_dir: 模型資料夾（含 tokens.txt 與 encoder / decoder / joiner）。
        encoder / decoder / joiner: 模型資料夾內的檔名。
        keywords: [{"tokens", "label"}, ...]，見 keywords_text()。
        keywords_threshold: 觸發門檻（0–1，越高越不容易誤喚醒，也越容易漏）；預設 0.25。
        keywords_score: 喚醒詞路徑的加分（越高越容易觸發）；預設 1.0。
        num_trailing_blanks: 喚醒詞後需要幾個空白幀才確認；預設 1。
        num_threads: 預設 1。
        chunk_ms: spot() 每次送入的長度（模擬麥克風串流）；預設 100。
    """

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        for key in ("model_dir", "encoder", "decoder", "joiner", "keywords"):
            if key not in self.config:
                raise ValueError(f"KeywordSpotter 設定缺少 {key}")
        keywords_text(self.config["keywords"])  # 提早檢查格式
        self._kws = None
        self._tmp: tempfile.TemporaryDirectory | None = None

    def load(self) -> None:
        """載入模型。

        例外:
            ImportError: 未安裝 sherpa-onnx（uv sync --extra wake）。
            FileNotFoundError: 模型檔不存在（下載方式見 library/kws/README.md）。
            ValueError: keywords 用到模型沒有的 token。
        """
        import sherpa_onnx

        c = self.config
        d = resolve_path(c["model_dir"])
        files = {k: d / c[k] for k in ("encoder", "decoder", "joiner")}
        for p in [d / "tokens.txt", *files.values()]:
            if not p.is_file():
                raise FileNotFoundError(f"喚醒詞模型檔不存在：{p}（下載方式見 library/kws/README.md）")
        missing = missing_tokens(c["keywords"], d / "tokens.txt")
        if missing:
            raise ValueError(f"{d.name} 沒有這些 token：{missing}")
        self._tmp = tempfile.TemporaryDirectory()
        kw_file = Path(self._tmp.name) / "keywords.txt"
        kw_file.write_text(keywords_text(c["keywords"]), encoding="utf-8")
        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=str(d / "tokens.txt"),
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
            keywords_file=str(kw_file),
            keywords_threshold=float(c.get("keywords_threshold", 0.25)),
            keywords_score=float(c.get("keywords_score", 1.0)),
            num_trailing_blanks=int(c.get("num_trailing_blanks", 1)),
            num_threads=int(c.get("num_threads", 1)),
            provider="cpu",
        )

    def stream(self, sr: int = 16000) -> KeywordStream:
        """建立一條新的串流（狀態獨立）。

        例外:
            RuntimeError: 尚未呼叫 load()。
        """
        if self._kws is None:
            raise RuntimeError("KeywordSpotter 尚未呼叫 load()")
        return KeywordStream(self._kws, self._kws.create_stream(), sr)

    def spot(self, wav: np.ndarray, sr: int) -> list[Detection]:
        """以 chunk_ms 為單位串流送入整段音訊（結尾補靜音），回傳所有喚醒。

        參數:
            wav: float32 一維陣列。
            sr: 取樣率（模型為 16 kHz；其他取樣率由 sherpa-onnx 內部重取樣）。

        回傳:
            Detection 列表，依時間排序。

        例外:
            RuntimeError: 尚未呼叫 load()。
        """
        s = self.stream(sr)
        step = max(1, round(sr * float(self.config.get("chunk_ms", 100)) / 1000))
        hits = []
        for i in range(0, len(wav), step):
            hits += s.accept(wav[i : i + step])
        hits += s.accept(np.zeros(round(_TAIL_SEC * sr), dtype=np.float32))
        return hits

    def unload(self) -> None:
        """釋放模型與暫存的 keywords 檔。"""
        self._kws = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
