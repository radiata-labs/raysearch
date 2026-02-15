from __future__ import annotations

import re
import unicodedata

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    return _WS_RE.sub(" ", t).strip()


def clean_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", (text or "")).strip()


def strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text or "").strip()


__all__ = ["clean_whitespace", "normalize_text", "strip_html"]
