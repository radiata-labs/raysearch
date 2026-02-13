from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from serpsage.settings.models import FetchSettings

_BINARY_PREFIXES = (
    "application/octet-stream",
    "application/zip",
    "application/x-",
    "image/",
    "video/",
    "audio/",
)
_HTML_CT_HINTS = ("text/html", "application/xhtml+xml")
_TEXT_CT_HINTS = ("text/plain", "text/markdown")
_PDF_CT_HINTS = ("application/pdf",)
_SPA_RE = re.compile(
    r"(id=[\"'](?:app|root|__next|__nuxt)[\"']|window\.__INITIAL_STATE__|"
    r"window\.__NUXT__|webpackJsonp|vite/client|reactroot|ng-version)",
    re.IGNORECASE,
)


def browser_headers(
    fetch_cfg: FetchSettings, *, profile: str | None = None
) -> dict[str, str]:
    ua = str(fetch_cfg.user_agent)

    headers: dict[str, str] = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    }

    if profile == "browser":
        headers.update(
            {
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            }
        )

    extra = fetch_cfg.extra_headers or {}
    for k, v in extra.items():
        if not k:
            continue
        headers[str(k)] = str(v)

    return headers


def get_delay_s(base_ms: int) -> float:
    return min(base_ms, 100) / 1000.0


def parse_retry_after_s(v: str | None) -> float | None:
    if not v:
        return None
    v = v.strip()
    if not v:
        return None
    if v.isdigit():
        return float(int(v))
    return None


def classify_content_kind(*, content_type: str | None, url: str, content: bytes) -> str:
    ct = (content_type or "").lower()
    path = (urlparse(url).path or "").lower()
    if path.endswith(".pdf") or any(h in ct for h in _PDF_CT_HINTS):
        return "pdf"
    if any(h in ct for h in _HTML_CT_HINTS):
        return "html"
    if any(h in ct for h in _TEXT_CT_HINTS):
        return "text"
    if ct and any(ct.startswith(pref) for pref in _BINARY_PREFIXES):
        return "binary"
    sample = (content or b"")[:2048].lstrip().lower()
    if sample.startswith(b"%pdf"):
        return "pdf"
    if b"<html" in sample or b"<!doctype" in sample:
        return "html"
    if b"\x00" in sample:
        return "binary"
    return "unknown"


def estimate_text_quality(content: bytes, *, content_kind: str) -> tuple[int, float, float]:
    if not content:
        return 0, 0.0, 1.0
    if content_kind in {"pdf", "binary"}:
        size = len(content)
        return size, min(1.0, size / 4096.0), 0.0

    sample = content[:300_000].decode("utf-8", errors="ignore")
    if content_kind == "text":
        txt = " ".join(sample.split())
        chars = len(txt)
        score = min(1.0, chars / 2400.0)
        return chars, score, 0.0

    # html
    try:
        soup = BeautifulSoup(sample, "html.parser")
        for t in soup.find_all(["script", "style", "noscript", "svg"]):
            t.decompose()
        txt = " ".join(soup.get_text(" ", strip=True).split())
        chars = len(txt)
        all_tags = max(1, len(soup.find_all(True)))
        scripts = len(soup.find_all("script"))
        script_ratio = float(scripts) / float(all_tags)
        links = " ".join(a.get_text(" ", strip=True) for a in soup.find_all("a"))
        link_chars = len(links)
        link_density = float(link_chars) / float(max(1, chars))
        score = min(
            1.0,
            (chars / 2200.0) * 0.75
            + (1.0 - min(1.0, link_density)) * 0.15
            + (1.0 - min(1.0, script_ratio)) * 0.10,
        )
        return chars, max(0.0, score), script_ratio
    except Exception:
        txt = " ".join(sample.split())
        chars = len(txt)
        score = min(1.0, chars / 2000.0)
        return chars, score, 0.0


def has_spa_signals(content: bytes) -> bool:
    sample = content[:60_000].decode("utf-8", errors="ignore")
    return bool(_SPA_RE.search(sample))


__all__ = [
    "browser_headers",
    "classify_content_kind",
    "estimate_text_quality",
    "get_delay_s",
    "has_spa_signals",
    "parse_retry_after_s",
]
