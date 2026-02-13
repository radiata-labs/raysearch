from __future__ import annotations

import re
from dataclasses import dataclass

from serpsage.text.normalize import clean_whitespace

_SENTENCE_BOUNDARY_RE = re.compile(r"([\u3002\uFF01\uFF1F!?;\uFF1B.\n])")
_LONG_SENT_SPLIT_RE = re.compile(r"([,\uFF0C\u3001\t ])")
_MD_LIST_RE = re.compile(r"^(\s*[-*+]\s+|\s*\d+[.)]\s+)")
_MD_HEAD_RE = re.compile(r"^(#{1,6})\s+")
_MD_TABLE_RE = re.compile(r"^\|.*\|$")


@dataclass(frozen=True, slots=True)
class MarkdownSegment:
    kind: str
    text: str


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


def markdown_to_segments(
    markdown: str,
    *,
    max_markdown_chars: int,
    max_segments: int,
    max_sentence_chars: int,
) -> list[MarkdownSegment]:
    text = (markdown or "").strip()
    if max_markdown_chars > 0 and len(text) > max_markdown_chars:
        text = text[:max_markdown_chars]
    if not text:
        return []

    lines = [ln.rstrip() for ln in text.splitlines()]
    segments: list[MarkdownSegment] = []
    in_code = False
    buf: list[str] = []
    buf_kind = "paragraph"

    def flush() -> None:
        nonlocal buf, buf_kind
        if not buf:
            return
        chunk = clean_whitespace(" ".join(buf))
        if chunk:
            if buf_kind == "paragraph" and len(chunk) > max_sentence_chars:
                for sent in split_sentences(chunk, max_sentence_chars=max_sentence_chars):
                    segments.append(MarkdownSegment(kind=buf_kind, text=sent))
            else:
                segments.append(MarkdownSegment(kind=buf_kind, text=chunk))
        buf = []
        buf_kind = "paragraph"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code:
                flush()
                in_code = True
                buf_kind = "code"
                buf = []
            else:
                flush()
                in_code = False
            continue
        if in_code:
            buf.append(line)
            continue

        if not stripped:
            flush()
            continue
        if _MD_HEAD_RE.match(stripped):
            flush()
            segments.append(MarkdownSegment(kind="heading", text=_MD_HEAD_RE.sub("", stripped)))
            continue
        if _MD_LIST_RE.match(stripped):
            flush()
            segments.append(MarkdownSegment(kind="list", text=_MD_LIST_RE.sub("", stripped)))
            continue
        if stripped.startswith(">"):
            flush()
            segments.append(MarkdownSegment(kind="quote", text=stripped.lstrip(">").strip()))
            continue
        if _MD_TABLE_RE.match(stripped):
            flush()
            segments.append(MarkdownSegment(kind="table", text=stripped.strip("| ").replace("|", " ")))
            continue
        if buf_kind != "paragraph":
            flush()
        buf_kind = "paragraph"
        buf.append(stripped)

    flush()
    if max_segments > 0:
        segments = segments[:max_segments]
    return [s for s in segments if s.text]


def chunk_segments(
    segments: list[MarkdownSegment],
    *,
    target_chars: int,
    overlap_segments: int,
    min_chunk_chars: int,
) -> list[str]:
    if not segments:
        return []
    target = max(1, int(target_chars))
    overlap = max(0, int(overlap_segments))

    chunks: list[str] = []
    cur: list[MarkdownSegment] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if not cur:
            return
        chunk = clean_whitespace(" ".join(seg.text for seg in cur))
        if len(chunk) >= int(min_chunk_chars):
            chunks.append(chunk)
        if overlap > 0:
            cur = cur[-overlap:]
            cur_len = sum(len(seg.text) + 1 for seg in cur)
        else:
            cur = []
            cur_len = 0

    for seg in segments:
        seg_text = (seg.text or "").strip()
        if not seg_text:
            continue
        if cur and cur_len + len(seg_text) + 1 > target:
            flush()
        cur.append(seg)
        cur_len += len(seg_text) + 1
    flush()
    return chunks


def chunk_sentences(
    sentences: list[str],
    *,
    target_chars: int,
    overlap_sentences: int,
    min_chunk_chars: int,
) -> list[str]:
    # Compatibility wrapper for old call sites.
    segments = [MarkdownSegment(kind="paragraph", text=s) for s in sentences if s]
    return chunk_segments(
        segments,
        target_chars=target_chars,
        overlap_segments=overlap_sentences,
        min_chunk_chars=min_chunk_chars,
    )


def prefilter_segments_by_tokens(
    segments: list[MarkdownSegment],
    *,
    query_tokens: list[str],
    min_hits: int,
    max_segments: int,
) -> list[MarkdownSegment]:
    if not segments:
        return []
    if not query_tokens:
        return segments[: max(1, int(max_segments))]
    wanted = max(1, int(min_hits))
    limited = max(1, int(max_segments))
    kept: list[MarkdownSegment] = []
    fallback: list[MarkdownSegment] = []
    lowered_tokens = [t.lower() for t in query_tokens if t]
    for seg in segments:
        txt = (seg.text or "").lower()
        if not txt:
            continue
        hits = 0
        for tok in lowered_tokens:
            if tok in txt:
                hits += 1
        if hits >= wanted:
            kept.append(seg)
        elif seg.kind in {"heading", "list"}:
            fallback.append(seg)
        if len(kept) >= limited:
            break
    if kept:
        return kept[:limited]
    return (fallback or segments)[:limited]


__all__ = [
    "MarkdownSegment",
    "chunk_segments",
    "chunk_sentences",
    "markdown_to_segments",
    "prefilter_segments_by_tokens",
    "split_sentences",
]
