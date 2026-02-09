from __future__ import annotations

import re

from serpsage.text.normalize import clean_whitespace

_SENTENCE_BOUNDARY_RE = re.compile(r"([\u3002\uFF01\uFF1F!?;\uFF1B.\n])")
_LONG_SENT_SPLIT_RE = re.compile(r"([,\uFF0C\u3001\t ])")


def split_sentences(text: str, *, max_sentence_chars: int) -> list[str]:
    cleaned = clean_whitespace(text or "")
    if not cleaned:
        return []
    parts = _SENTENCE_BOUNDARY_RE.split(cleaned)
    out: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        buf += part
        if _SENTENCE_BOUNDARY_RE.fullmatch(part):
            s = buf.strip()
            if s:
                out.extend(_split_long_sentence(s, max_sentence_chars))
            buf = ""
    tail = buf.strip()
    if tail:
        out.extend(_split_long_sentence(tail, max_sentence_chars))
    return [s for s in out if s]


def _split_long_sentence(sentence: str, max_len: int) -> list[str]:
    if max_len <= 0 or len(sentence) <= max_len:
        return [sentence]
    parts = _LONG_SENT_SPLIT_RE.split(sentence)
    out: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if len(buf) + len(part) > max_len and buf.strip():
            out.append(buf.strip())
            buf = ""
        buf += part
    if buf.strip():
        out.append(buf.strip())
    return out or [sentence]


def chunk_sentences(
    sentences: list[str],
    *,
    target_chars: int,
    overlap_sentences: int,
    min_chunk_chars: int,
) -> list[str]:
    if not sentences:
        return []
    target = max(1, int(target_chars))
    overlap = max(0, int(overlap_sentences))

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        chunk = clean_whitespace(" ".join(cur))
        if len(chunk) >= int(min_chunk_chars):
            chunks.append(chunk)
        if overlap > 0:
            cur = cur[-overlap:]
            cur_len = sum(len(s) + 1 for s in cur)
        else:
            cur = []
            cur_len = 0

    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue
        if cur and cur_len + len(s) + 1 > target:
            flush()
        cur.append(s)
        cur_len += len(s) + 1
    flush()
    return chunks


__all__ = ["chunk_sentences", "split_sentences"]
