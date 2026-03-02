from __future__ import annotations

import codecs
import re
from contextlib import suppress
from typing import Literal

ContentKind = Literal["html", "text"]
_WS_RE = re.compile(r"\s+")
_META_CHARSET_RE = re.compile(
    r"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)
_CT_CHARSET_RE = re.compile(r"charset\s*=\s*([a-zA-Z0-9_\-]+)", re.IGNORECASE)


def looks_like_html(sample: bytes) -> bool:
    head = sample[:8192].lower()
    return any(
        tok in head
        for tok in (b"<!doctype", b"<html", b"<meta", b"<body", b"</p", b"</div")
    )


def extract_meta_charset(data: bytes) -> str | None:
    head = data[:16384].decode("ascii", errors="ignore")
    m = _META_CHARSET_RE.search(head)
    if not m:
        return None
    cs = (m.group(1) or "").strip()
    return cs or None


def guess_apparent_encoding(data: bytes) -> str | None:
    sample = data[:65536]
    with suppress(Exception):
        from charset_normalizer import from_bytes  # noqa: I001, PLC0415

        best = from_bytes(sample).best()
        enc = getattr(best, "encoding", None) if best is not None else None
        if enc:
            return str(enc)
    with suppress(Exception):
        import chardet  # noqa: PLC0415

        det = chardet.detect(sample)
        enc = det.get("encoding") if isinstance(det, dict) else None
        if enc:
            return str(enc)
    return None


def decode_best_effort(
    data: bytes,
    *,
    content_type: str | None,
    resp_encoding: str | None = None,
    apparent_encoding: str | None = None,
) -> tuple[str, ContentKind]:
    if not data:
        return "", "text"
    kind: ContentKind = "html" if looks_like_html(data) else "text"
    candidates: list[str] = []
    declared: list[str] = []
    if data.startswith(codecs.BOM_UTF8):
        candidates.append("utf-8-sig")
        declared.append("utf-8-sig")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        candidates.append("utf-16")
        declared.append("utf-16")
    if content_type:
        m = _CT_CHARSET_RE.search(content_type)
        if m:
            cs = m.group(1)
            candidates.append(cs)
            declared.append(cs)
    meta = extract_meta_charset(data)
    if meta:
        candidates.append(meta)
        declared.append(meta)
    if resp_encoding:
        candidates.append(resp_encoding)
    if apparent_encoding:
        candidates.append(apparent_encoding)
    candidates.extend(
        [
            "utf-8",
            "utf-8-sig",
            "gb18030",
            "shift_jis",
            "euc_jp",
            "iso-2022-jp",
            "cp1252",
            "latin-1",
        ]
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        c = (c or "").strip()
        if not c:
            continue
        lc = c.lower()
        if lc in seen:
            continue
        seen.add(lc)
        ordered.append(c)
    # Prefer declared charset when it produces "clean" output.
    for enc in declared:
        try:
            text = data.decode(enc, errors="replace")
        except Exception:  # noqa: S112
            continue
        text = text.replace("\x00", "")
        total = max(1, len(text))
        repl = text.count("\ufffd") / total
        ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t") / total
        if repl <= 0.001 and ctrl <= 0.001:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            if kind == "text":
                text = _WS_RE.sub(" ", text)
            return text, kind
    best_text = ""
    best_key: tuple[float, float, float, float] | None = None
    for enc in ordered:
        try:
            text = data.decode(enc, errors="replace")
        except Exception:  # noqa: S112
            continue
        text = text.replace("\x00", "")
        total = max(1, len(text))
        repl = text.count("\ufffd") / total
        ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t") / total
        cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", text)) / total
        short_penalty = 1.0 if len(text) < 200 else 0.0
        key = (repl, ctrl, -cjk, short_penalty)
        if best_key is None or key < best_key:
            best_key = key
            best_text = text
    if not best_text:
        best_text = data.decode("utf-8", errors="replace").replace("\x00", "")
    best_text = best_text.replace("\r\n", "\n").replace("\r", "\n")
    if kind == "text":
        best_text = _WS_RE.sub(" ", best_text)
    return best_text, kind


__all__ = [
    "ContentKind",
    "decode_best_effort",
    "guess_apparent_encoding",
]
