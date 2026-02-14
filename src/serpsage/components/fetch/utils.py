from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from bs4 import BeautifulSoup

DEFAULT_USER_AGENT = "serpsage-bot/4.0"

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
    r"window\.__NUXT__|webpackJsonp|vite/client|reactroot|ng-version|"
    r"data-reactroot|hydration|astro-island|sveltekit|chunk-vendors)",
    re.IGNORECASE,
)


def browser_headers(
    *,
    profile: str | None = None,
    user_agent: str | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": user_agent or DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
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
    return headers


def get_delay_s(base_ms: int) -> float:
    return min(max(0, int(base_ms)), 250) / 1000.0


def parse_retry_after_s(v: str | None) -> float | None:
    if not v:
        return None
    raw = v.strip()
    if not raw:
        return None
    if raw.isdigit():
        return float(int(raw))
    return None


def classify_content_kind(
    *, content_type: str | None, url: str, content: bytes
) -> Literal["html", "pdf", "text", "binary", "unknown"]:
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


def estimate_text_quality(
    content: bytes, *, content_kind: str
) -> tuple[int, float, float]:
    if not content:
        return 0, 0.0, 1.0
    if content_kind in {"pdf", "binary"}:
        size = len(content)
        return size, min(1.0, size / 4096.0), 0.0

    sample = content[:450_000].decode("utf-8", errors="ignore")
    if content_kind == "text":
        txt = " ".join(sample.split())
        chars = len(txt)
        score = min(1.0, chars / 2600.0)
        return chars, score, 0.0

    try:
        soup = BeautifulSoup(sample, "html.parser")
        script_tags = soup.find_all("script")
        for t in soup.find_all(["script", "style", "noscript", "svg"]):
            t.decompose()
        txt = " ".join(soup.get_text(" ", strip=True).split())
        chars = len(txt)
        if chars <= 0:
            return 0, 0.0, 1.0
        all_tags = max(1, len(soup.find_all(True)))
        script_ratio = float(len(script_tags)) / float(max(1, all_tags))
        links = " ".join(a.get_text(" ", strip=True) for a in soup.find_all("a"))
        link_chars = len(links)
        link_density = float(link_chars) / float(max(1, chars))
        punct = len(re.findall(r"[,.!?;:\u3002\uff01\uff1f\uff1b]", txt))
        punct_density = float(punct) / float(max(1, chars))
        score = min(
            1.0,
            (chars / 2500.0) * 0.72
            + (1.0 - min(1.0, link_density)) * 0.16
            + min(1.0, punct_density * 90.0) * 0.07
            + (1.0 - min(1.0, script_ratio)) * 0.05,
        )
        return chars, max(0.0, score), script_ratio
    except Exception:
        txt = " ".join(sample.split())
        chars = len(txt)
        score = min(1.0, chars / 2200.0)
        return chars, score, 0.0


def has_spa_signals(content: bytes) -> bool:
    sample = content[:80_000].decode("utf-8", errors="ignore")
    return bool(_SPA_RE.search(sample))


def blocked_marker_hit(
    content: bytes, *, markers: tuple[str, ...] | list[str] | None = None
) -> bool:
    if not content:
        return False
    use_markers = tuple(
        marker.strip().lower() for marker in (markers or ()) if marker and marker.strip()
    )
    if not use_markers:
        return False

    raw_sample = content[:30_000].decode("utf-8", errors="ignore")
    lowered = raw_sample.lower()
    looks_like_html = bool(
        "<html" in lowered
        or "<!doctype" in lowered
        or "<body" in lowered
        or "<head" in lowered
    )

    if looks_like_html:
        try:
            soup = BeautifulSoup(raw_sample, "html.parser")
            for t in soup.find_all(["script", "style", "noscript"]):
                t.decompose()
            visible = " ".join(soup.get_text(" ", strip=True).split()).lower()
            if visible:
                return any(marker in visible for marker in use_markers)
        except Exception:
            pass

    return any(marker in lowered for marker in use_markers)


__all__ = [
    "blocked_marker_hit",
    "browser_headers",
    "classify_content_kind",
    "estimate_text_quality",
    "get_delay_s",
    "has_spa_signals",
    "parse_retry_after_s",
]
