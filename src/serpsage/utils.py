from __future__ import annotations

import json
import re
import unicodedata
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Iterable

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TRACKING_QUERY_KEYS = {"gclid", "fbclid", "msclkid"}
_TRACKING_QUERY_PREFIXES = ("utm_",)


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


def canonicalize_url(raw_url: str) -> str:
    token = clean_whitespace(raw_url)
    if not token:
        return ""
    try:
        parsed = urlsplit(token)
    except Exception:  # noqa: BLE001
        return token
    scheme = clean_whitespace(parsed.scheme).lower() or "https"
    host = clean_whitespace(str(parsed.hostname or "")).lower()
    if not host:
        return token
    port = _resolve_port(parsed)
    netloc = _compose_netloc(scheme=scheme, host=host, port=port)
    path = clean_whitespace(parsed.path) or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/":
        path = path.rstrip("/") or "/"
    pairs: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        normalized_key = clean_whitespace(key)
        if not normalized_key:
            continue
        key_lc = normalized_key.casefold()
        if key_lc in _TRACKING_QUERY_KEYS:
            continue
        if any(key_lc.startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES):
            continue
        pairs.append((normalized_key, clean_whitespace(value)))
    pairs.sort(key=lambda item: (item[0].casefold(), item[1]))
    query = urlencode(pairs, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def _resolve_port(parsed: Any) -> int | None:
    try:
        value = parsed.port
    except Exception:  # noqa: BLE001
        return None
    return int(value) if value is not None else None


def _compose_netloc(*, scheme: str, host: str, port: int | None) -> str:
    if port is None:
        return host
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        return host
    return f"{host}:{int(port)}"


__all__ = [
    "canonicalize_url",
    "clean_whitespace",
    "normalize_text",
    "strip_html",
    "uniq_preserve_order",
    "stable_json",
]
