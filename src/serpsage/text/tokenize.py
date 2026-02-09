from __future__ import annotations

import re

from serpsage.text.normalize import normalize_text
from serpsage.util.collections import uniq_preserve_order

WORD_RE = re.compile(r"[A-Za-z0-9]+")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]+")

try:
    import jieba  # type: ignore[import-not-found]

    JIEBA_AVAILABLE = True
except Exception:  # noqa: BLE001
    jieba = None
    JIEBA_AVAILABLE = False


def ngrams(text: str, n: int) -> list[str]:
    if n <= 0 or len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def tokenize(text: str) -> list[str]:
    t = normalize_text(text)
    if not t:
        return []

    tokens: list[str] = []
    tokens.extend(m.group(0).lower() for m in WORD_RE.finditer(t))

    if JIEBA_AVAILABLE and jieba is not None:
        for tok in jieba.lcut(t, cut_all=False):
            tok = tok.strip()
            if not tok:
                continue
            # Filter extremely short tokens.
            if len(tok) == 1 and CJK_RUN_RE.fullmatch(tok):
                continue
            if CJK_RUN_RE.fullmatch(tok):
                tokens.append(tok)
            elif WORD_RE.fullmatch(tok):
                tokens.append(tok.lower())
    else:
        # Fallback: take CJK runs directly.
        for run in CJK_RUN_RE.findall(t):
            if len(run) >= 2:
                tokens.append(run)  # noqa: PERF401

    # Add CJK n-grams as recall boost.
    for run in CJK_RUN_RE.findall(t):
        if len(run) <= 3:
            tokens.append(run)
        else:
            tokens.extend(ngrams(run, 2))
            tokens.extend(ngrams(run, 3))

    return uniq_preserve_order(tokens)


__all__ = ["tokenize"]
