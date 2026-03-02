from __future__ import annotations

import json
import re
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def uniq_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def stable_json(obj: Any) -> str:
    """Deterministic JSON representation used for cache keys."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


__all__ = [
    "clean_whitespace",
    "normalize_text",
    "strip_html",
    "uniq_preserve_order",
    "stable_json",
]
