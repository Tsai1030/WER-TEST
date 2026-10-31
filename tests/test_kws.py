"""喚醒詞：keywords 格式、token 檢查；有下載模型時另測實際偵測（模型的附帶測試音檔）。"""

from pathlib import Path

import numpy as np
import pytest

from library.kws import KeywordSpotter, keywords_text, missing_tokens
from library.utils.paths import resolve_path

KW = [{"tokens": "x iǎo m ī x iǎo m ī", "label": "小咪小咪"}]
ZH_EN = resolve_path(".kws_models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
TAG = "epoch-13-avg-2-chunk-16-left-64"


def test_keywords_text_format_and_errors() -> None:
    assert keywords_text(KW) == "x iǎo m ī x iǎo m ī @小咪小咪\n"
    with pytest.raises(ValueError):
        keywords_text([])
    with pytest.raises(ValueError):
        keywords_text([{"tokens": "", "label": "a"}])
    with pytest.raises(ValueError):
        keywords_text(KW + KW)


def test_missing_tokens(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blk> 0\nx 1\niǎo 2\nm 3\n", encoding="utf-8")
    assert missing_tokens(KW, tokens) == ["ī"]
    with pytest.raises(FileNotFoundError):
        missing_tokens(KW, tmp_path / "none.txt")


def test_spotter_requires_load_and_fields() -> None:
    with pytest.raises(ValueError):
        KeywordSpotter({"keywords": KW})
    s = KeywordSpotter({"model_dir": "x", "encoder": "e", "decoder": "d", "joiner": "j", "keywords": KW})
    with pytest.raises(RuntimeError):
        s.stream()


@pytest.mark.skipif(not ZH_EN.is_dir(), reason="未下載喚醒詞模型（見 library/kws/README.md）")
def test_spotter_detects_bundled_keyword() -> None:
    import soundfile as sf

    cfg = {"model_dir": str(ZH_EN), "encoder": f"encoder-{TAG}.onnx", "decoder": f"decoder-{TAG}.onnx",
           "joiner": f"joiner-{TAG}.onnx", "keywords": [{"tokens": "L AY1 T AH1 P", "label": "LIGHT_UP"}, *KW]}
    s = KeywordSpotter(cfg)
    s.load()
    try:
        wav, sr = sf.read(ZH_EN / "test_wavs" / "en_0.wav", dtype="float32")
        hits = s.spot(wav, sr)
        assert [h.keyword for h in hits] == ["LIGHT_UP"]
        assert 0 < hits[0].t_sec <= len(wav) / sr + 1
        assert s.spot(np.zeros(sr * 3, dtype=np.float32), sr) == []
    finally:
        s.unload()
