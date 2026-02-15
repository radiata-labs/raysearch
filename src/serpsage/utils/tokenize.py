from __future__ import annotations

import re

from serpsage.utils.collections import uniq_preserve_order
from serpsage.utils.normalize import normalize_text
from serpsage.utils.stopwords import is_stopword

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


def tokenize_for_query(query: str) -> list[str]:
    items = tokenize(query)

    def to_zh_tw(s: str) -> str:
        if OPENCC_AVAILABLE and opencc is not None:
            return opencc.OpenCC("s2t").convert(s)
        try:
            import zhconv  # type: ignore[import-not-found]
        except Exception:
            return s

        return zhconv.convert(s, "zh-tw")

    merged = items + [to_zh_tw(x) for x in items]

    seen: set[str] = set()
    out: list[str] = []
    for x in merged:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


__all__ = ["tokenize", "tokenize_for_query"]
