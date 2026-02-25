from __future__ import annotations

import random
import re
from typing import Literal
from urllib.parse import urlparse

from bs4 import BeautifulSoup

DEFAULT_USER_AGENT = "serpsage-bot/4.0"

# Real browser User-Agents for rotation (updated for 2024-2025)
USER_AGENTS = [
    # Chrome 124 on Windows 11
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome 123 on Windows 10
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Chrome 124 on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome 124 on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox 124 on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    # Firefox 124 on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
    # Edge 123 on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    # Safari 17 on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

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
_NEXTJS_RE = re.compile(
    r"(id=[\"']__next[\"']|__NEXT_DATA__|/_next/static/|/_next/image(?:\?|/)|"
    r"next-route-announcer|next-size-adjust|data-nextjs)",
    re.IGNORECASE,
)


def get_random_user_agent() -> str:
    """Return a random real browser User-Agent string."""
    return random.choice(USER_AGENTS)


def browser_headers(
    *,
    profile: str | None = None,
    user_agent: str | None = None,
    randomize: bool = True,
) -> dict[str, str]:
    """Generate browser-like HTTP headers.

    Args:
        profile: Profile type (e.g., "browser" for full browser headers)
        user_agent: Custom user agent. If None and randomize=True, uses random UA.
        randomize: If True and user_agent is None, uses a random real browser UA.

    Returns:
        Dictionary of HTTP headers.
    """
    # Determine user agent
    if user_agent is not None:
        ua = user_agent
    elif randomize:
        ua = get_random_user_agent()
    else:
        ua = DEFAULT_USER_AGENT

    headers: dict[str, str] = {
        "User-Agent": ua,
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

    # PDF detection (highest priority)
    if path.endswith(".pdf") or any(h in ct for h in _PDF_CT_HINTS):
        return "pdf"

    # HTML detection from content-type header
    if any(h in ct for h in _HTML_CT_HINTS):
        return "html"

    # Plain text detection from content-type header
    if any(h in ct for h in _TEXT_CT_HINTS):
        return "text"

    # Binary detection from content-type header
    if ct and any(ct.startswith(pref) for pref in _BINARY_PREFIXES):
        return "binary"

    # Content-based detection (when header is missing or ambiguous)
    sample = (content or b"")[:2048].lstrip()
    sample_lower = sample.lower()

    # PDF magic bytes
    if sample.startswith((b"%pdf", b"%PDF")):
        return "pdf"

    # HTML detection - enhanced with more patterns
    html_patterns = [
        b"<html",
        b"<!doctype",
        b"<head",
        b"<body",
        b"<!DOCTYPE",
        b"<HTML",
        b"<HEAD",
        b"<BODY",
        b"<meta",
        b"<title",
        b"<div",
        b"<article",
        b"<?xml",
        b"<xhtml",
    ]
    if any(pattern in sample_lower or pattern in sample for pattern in html_patterns):
        return "html"

    # Binary detection - null bytes or non-printable chars
    if b"\x00" in sample:
        return "binary"

    # Check for high ratio of non-printable characters
    try:
        text_sample = sample.decode("utf-8", errors="ignore")
        if text_sample:
            non_printable = sum(
                1 for c in text_sample if not c.isprintable() and c not in "\n\r\t"
            )
            if len(text_sample) > 0 and non_printable / len(text_sample) > 0.3:
                return "binary"
    except Exception:
        pass

    # Default to unknown for ambiguous content
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


def has_nextjs_signals(content: bytes) -> bool:
    sample = content[:120_000].decode("utf-8", errors="ignore")
    return bool(_NEXTJS_RE.search(sample))


def blocked_marker_hit(
    content: bytes, *, markers: tuple[str, ...] | list[str] | None = None
) -> bool:
    """Check if content contains blocked markers with context awareness.

    This function distinguishes between:
    - Actual blocking pages (Cloudflare challenge, access denied)
    - Technical content that mentions blocking services (e.g., tutorials about Cloudflare)

    A marker hit is only considered a block if:
    1. The marker appears in the title/heading
    2. The marker appears in the first 500 characters (prominent position)
    3. Multiple markers appear together (stronger signal)
    """
    if not content:
        return False
    use_markers = tuple(
        marker.strip().lower()
        for marker in (markers or ())
        if marker and marker.strip()
    )
    if not use_markers:
        return False

    raw_sample = content[:30_000].decode("utf-8", errors="ignore")
    lowered = raw_sample.lower()

    # Check if content looks like HTML
    looks_like_html = bool(
        "<html" in lowered
        or "<!doctype" in lowered
        or "<body" in lowered
        or "<head" in lowered
    )

    # Extract visible text from HTML
    visible_text = lowered
    title_text = ""
    heading_text = ""

    if looks_like_html:
        try:
            soup = BeautifulSoup(raw_sample, "html.parser")

            # Extract title for special checking
            title_tag = soup.find("title")
            if title_tag:
                title_text = title_tag.get_text(" ", strip=True).lower()

            # Extract headings (h1, h2) for special checking
            for h in soup.find_all(["h1", "h2"]):
                heading_text += " " + h.get_text(" ", strip=True).lower()

            # Remove script, style, noscript for visible text
            for t in soup.find_all(["script", "style", "noscript"]):
                t.decompose()
            visible_text = " ".join(soup.get_text(" ", strip=True).split()).lower()

            if not visible_text:
                # Fall back to raw sample if no visible text
                visible_text = lowered
        except Exception:
            pass

    # Count total marker hits
    total_hits = sum(1 for marker in use_markers if marker in visible_text)
    if total_hits == 0:
        return False

    # Check for "strong signals" - markers in title or headings
    strong_signals = sum(
        1 for marker in use_markers if marker in title_text or marker in heading_text
    )

    # Check for "weak signals" - markers only in body content (likely technical mention)
    weak_only_signals = sum(
        1
        for marker in use_markers
        if marker in visible_text
        and marker not in title_text
        and marker not in heading_text
    )

    # Decision logic:
    # 1. If marker in title + heading = definite block
    if strong_signals >= 2:
        return True

    # 2. If marker in title OR heading + at least one more hit = likely block
    if strong_signals >= 1 and total_hits >= 2:
        return True

    # 3. If only weak signals and content is long (>2000 chars), likely just technical mention
    if weak_only_signals == total_hits and len(visible_text) > 2000:
        return False

    # 4. If marker only appears once in visible text, consider it a technical mention
    if total_hits == 1 and strong_signals == 0:
        return False

    # 5. Multiple weak signals still indicates blocking
    if total_hits >= 2:
        return True

    # Default: single hit in visible text
    return total_hits >= 1


__all__ = [
    "blocked_marker_hit",
    "browser_headers",
    "classify_content_kind",
    "estimate_text_quality",
    "get_delay_s",
    "get_random_user_agent",
    "has_nextjs_signals",
    "has_spa_signals",
    "parse_retry_after_s",
    "USER_AGENTS",
]
