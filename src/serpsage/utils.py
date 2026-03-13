from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Iterable

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ISO8601_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
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


def parse_iso8601_datetime(value: str) -> datetime | None:
    token = clean_whitespace(value)
    if not token:
        return None
    normalized = token
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        if _ISO8601_DATE_ONLY_RE.fullmatch(normalized):
            parsed = datetime.fromisoformat(f"{normalized}T00:00:00")
        else:
            parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def is_iso8601_date_only(value: str) -> bool:
    return _ISO8601_DATE_ONLY_RE.fullmatch(clean_whitespace(value)) is not None


def normalize_iso8601_string(value: str, *, allow_blank: bool = False) -> str:
    token = clean_whitespace(value)
    if not token:
        if allow_blank:
            return ""
        raise ValueError("value must not be empty")
    parsed = parse_iso8601_datetime(token)
    if parsed is None:
        raise ValueError("value must be a valid ISO 8601 string")
    if is_iso8601_date_only(token):
        return parsed.date().isoformat()
    return parsed.isoformat()


def iso8601_start_date_floor(value: str) -> str:
    parsed = parse_iso8601_datetime(value)
    if parsed is None:
        return ""
    return parsed.date().isoformat()


def iso8601_end_date_exclusive(value: str) -> str:
    parsed = parse_iso8601_datetime(value)
    if parsed is None:
        return ""
    return (parsed.date() + timedelta(days=1)).isoformat()


def published_date_in_range(
    published_date: str,
    *,
    start_published_date: str = "",
    end_published_date: str = "",
) -> bool:
    if not clean_whitespace(start_published_date) and not clean_whitespace(
        end_published_date
    ):
        return True
    published_at = parse_iso8601_datetime(published_date)
    if published_at is None:
        return False
    start_token = clean_whitespace(start_published_date)
    if start_token:
        start_at = parse_iso8601_datetime(start_token)
        if start_at is None or published_at < start_at:
            return False
    end_token = clean_whitespace(end_published_date)
    if not end_token:
        return True
    end_at = parse_iso8601_datetime(end_token)
    if end_at is None:
        return False
    if is_iso8601_date_only(end_token):
        return published_at < (end_at + timedelta(days=1))
    return published_at <= end_at


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
    "is_iso8601_date_only",
    "iso8601_end_date_exclusive",
    "iso8601_start_date_floor",
    "normalize_text",
    "normalize_iso8601_string",
    "parse_iso8601_datetime",
    "published_date_in_range",
    "strip_html",
    "uniq_preserve_order",
    "stable_json",
]
