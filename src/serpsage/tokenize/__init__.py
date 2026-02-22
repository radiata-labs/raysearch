from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, cast

from serpsage.tokenize.stopwords import is_stopword
from serpsage.utils import normalize_text, uniq_preserve_order

WORD_RE = re.compile(r"[A-Za-z0-9]+")
CJK_KANA_RUN_RE = re.compile(
    r"[\u3400-\u9fff\u3040-\u30ff\u31f0-\u31ff\u3005\u30fc\uff66-\uff9d]+"
)

KEEP_POS_ZH = {
    "n",
    "nr",
    "nr1",
    "nr2",
    "nrj",
    "nrf",
    "ns",
    "nsf",
    "nt",
    "nz",
    "nl",
    "ng",
    "v",
    "vd",
    "vn",
    "a",
    "ad",
    "an",
    "i",
    "l",
    "j",
}

KEEP_POS_JA = {"名詞", "動詞", "形容詞", "副詞", "記号", "接頭詞", "接尾辞"}

try:
    import jieba.posseg as pseg  # type: ignore[import-untyped]

    JIEBA_AVAILABLE = True
except Exception:  # noqa: BLE001
    pseg = None
    JIEBA_AVAILABLE = False

try:
    from sudachipy import (  # type: ignore[import-untyped]
        dictionary as sudachi_dictionary,
    )
    from sudachipy import tokenizer as sudachi_tokenizer  # type: ignore[import-untyped]

    sudachi = sudachi_dictionary.Dictionary().create()
    sudachi_mode = sudachi_tokenizer.Tokenizer.SplitMode.A
    SUDACHI_AVAILABLE = True
except Exception:  # noqa: BLE001
    sudachi = None
    sudachi_mode = None
    SUDACHI_AVAILABLE = False

try:
    import opencc  # type: ignore[import-untyped]

    OPENCC_AVAILABLE = True
except Exception:  # noqa: BLE001
    opencc = None
    OPENCC_AVAILABLE = False


class _ConverterCache:
    """Cache for lazy-loaded converter modules."""

    _opencc_s2t_converter: object | None = None
    _zhconv_module: object | None = None
    _zhconv_unavailable = False


def _has_kana(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if (
            (0x3040 <= o <= 0x30FF)
            or (0x31F0 <= o <= 0x31FF)
            or (0xFF66 <= o <= 0xFF9D)
        ):
            return True
    return False


@lru_cache(maxsize=1024)
def tokenize(text: str) -> list[str]:
    t = normalize_text(text)
    if not t:
        return []

    tokens: list[str] = []
    tokens.extend(m.group(0).lower() for m in WORD_RE.finditer(t))

    if JIEBA_AVAILABLE and pseg is not None:
        for tok, flag in pseg.cut(t, use_paddle=True):
            tok = tok.strip()
            if not tok:
                continue
            if flag not in KEEP_POS_ZH:
                continue
            if is_stopword(tok):
                continue
            if CJK_KANA_RUN_RE.fullmatch(tok):
                tokens.append(tok)
            elif WORD_RE.fullmatch(tok):
                tokens.append(tok.lower())

    if (
        SUDACHI_AVAILABLE
        and sudachi is not None
        and sudachi_mode is not None
        and _has_kana(t)
    ):
        for m in sudachi.tokenize(t, sudachi_mode):
            tok, pos = m.surface().strip(), m.part_of_speech()
            if not tok:
                continue
            if WORD_RE.fullmatch(tok):
                tok = tok.lower()
            if is_stopword(tok):
                continue
            if pos and not any(p in KEEP_POS_JA for p in pos):
                continue
            if not CJK_KANA_RUN_RE.fullmatch(tok) and not WORD_RE.fullmatch(tok):
                continue
            tokens.append(tok)

    if (not JIEBA_AVAILABLE or pseg is None) and (
        not SUDACHI_AVAILABLE or sudachi is None
    ):
        for run in CJK_KANA_RUN_RE.findall(t):
            run = run.strip()
            if len(run) < 2:
                continue
            tokens.append(run)

    return uniq_preserve_order(tokens)


@lru_cache(maxsize=1024)
def tokenize_for_query(query: str) -> list[str]:
    items = tokenize(query)
    merged = items + [_to_zh_tw(x) for x in items]

    seen: set[str] = set()
    out: list[str] = []
    for x in merged:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _to_zh_tw(text: str) -> str:
    converter = _get_opencc_s2t_converter()
    if converter is not None:
        try:
            return str(cast("Any", converter).convert(text))
        except Exception:  # noqa: BLE001
            return text

    zhconv_module = _get_zhconv_module()
    if zhconv_module is None:
        return text
    try:
        return str(cast("Any", zhconv_module).convert(text, "zh-tw"))
    except Exception:  # noqa: BLE001
        return text


def _get_opencc_s2t_converter() -> object | None:
    if not OPENCC_AVAILABLE or opencc is None:
        return None
    if _ConverterCache._opencc_s2t_converter is None:
        try:
            _ConverterCache._opencc_s2t_converter = opencc.OpenCC("s2t")
        except Exception:  # noqa: BLE001
            return None
    return _ConverterCache._opencc_s2t_converter


def _get_zhconv_module() -> object | None:
    if _ConverterCache._zhconv_unavailable:
        return None
    if _ConverterCache._zhconv_module is not None:
        return _ConverterCache._zhconv_module
    try:
        import zhconv as _zhconv  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        _ConverterCache._zhconv_unavailable = True
        return None
    _ConverterCache._zhconv_module = _zhconv
    return _ConverterCache._zhconv_module


__all__ = ["tokenize", "tokenize_for_query"]
