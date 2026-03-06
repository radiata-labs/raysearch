from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast
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
_PUNCT_RE = re.compile(r"[,.!?;:\u3002\uff01\uff1f\uff1b]")
_SPACE_RE = re.compile(r"\s+")
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
_UUID_SEGMENT_RE = re.compile(
    r"^[0-9a-f]{8,}(?:-[0-9a-f]{4,}){1,}[0-9a-f]{4,}$",
    re.IGNORECASE,
)
_INT_SEGMENT_RE = re.compile(r"^\d+$")
_HASH_SEGMENT_RE = re.compile(r"^[0-9a-f]{12,}$", re.IGNORECASE)
if TYPE_CHECKING:
    from selectolax.parser import HTMLParser as _SelectolaxHTMLParser

    _SELECTOLAX_AVAILABLE = True
else:
    try:
        from selectolax.parser import (
            HTMLParser as _SelectolaxHTMLParser,  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
        )

        _SELECTOLAX_AVAILABLE = True
    except Exception:  # noqa: BLE001
        _SelectolaxHTMLParser = None
        _SELECTOLAX_AVAILABLE = False


@dataclass(slots=True)
class HtmlSignals:
    text_chars: int
    script_ratio: float
    link_density: float
    punctuation_density: float
    visible_text: str
    title_text: str
    heading_text: str
    tag_count: int
    parser_name: Literal["selectolax", "beautifulsoup", "fallback"]


@dataclass(slots=True)
class ContentAnalysis:
    content_kind: Literal["html", "pdf", "text", "binary", "unknown"]
    text_chars: int
    content_score: float
    script_ratio: float
    blocked: bool
    nextjs: bool
    spa: bool
    html_signals: HtmlSignals | None = None


def get_random_user_agent() -> str:
    """Return a random real browser User-Agent string."""
    return random.choice(USER_AGENTS)


def browser_headers(
    *,
    profile: str | None = None,
    user_agent: str | None = None,
    randomize: bool = True,
) -> dict[str, str]:
    """Generate browser-like HTTP headers."""
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


def normalize_route_key(url: str) -> str:
    parsed = urlparse(url)
    host = str(parsed.netloc or "").lower()
    raw_parts = [part for part in str(parsed.path or "/").split("/") if part]
    if not raw_parts:
        return f"{host}/"
    normalized_parts = [_normalize_path_segment(part) for part in raw_parts[:4]]
    return f"{host}/{'/'.join(normalized_parts)}"


def _normalize_path_segment(part: str) -> str:
    token = part.strip().lower()
    if not token:
        return "_"
    if _INT_SEGMENT_RE.fullmatch(token):
        return ":int"
    if _UUID_SEGMENT_RE.fullmatch(token):
        return ":uuid"
    if _HASH_SEGMENT_RE.fullmatch(token):
        return ":hash"
    if len(token) > 40:
        return ":long"
    return token


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
    sample = (content or b"")[:2048].lstrip()
    sample_lower = sample.lower()
    if sample.startswith((b"%pdf", b"%PDF")):
        return "pdf"
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
    if b"\x00" in sample:
        return "binary"
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
    return "unknown"


def analyze_content(
    *,
    content: bytes,
    content_type: str | None,
    url: str,
    markers: tuple[str, ...] | list[str] | None = None,
) -> ContentAnalysis:
    content_kind = classify_content_kind(
        content_type=content_type,
        url=url,
        content=content,
    )
    html_signals: HtmlSignals | None = None
    if content_kind == "html":
        html_signals = extract_html_signals(content)
        text_chars = int(html_signals.text_chars)
        script_ratio = float(html_signals.script_ratio)
        score = min(
            1.0,
            (text_chars / 2500.0) * 0.72
            + (1.0 - min(1.0, html_signals.link_density)) * 0.16
            + min(1.0, html_signals.punctuation_density * 90.0) * 0.07
            + (1.0 - min(1.0, script_ratio)) * 0.05,
        )
        blocked = blocked_marker_hit(
            content,
            markers=markers,
            html_signals=html_signals,
        )
        return ContentAnalysis(
            content_kind=content_kind,
            text_chars=text_chars,
            content_score=max(0.0, score),
            script_ratio=script_ratio,
            blocked=blocked,
            nextjs=has_nextjs_signals(content),
            spa=has_spa_signals(content),
            html_signals=html_signals,
        )
    if content_kind in {"pdf", "binary"}:
        size = len(content)
        return ContentAnalysis(
            content_kind=content_kind,
            text_chars=size,
            content_score=min(1.0, size / 4096.0),
            script_ratio=0.0,
            blocked=False,
            nextjs=False,
            spa=False,
            html_signals=None,
        )
    sample = (content or b"")[:450_000].decode("utf-8", errors="ignore")
    txt = _collapse_whitespace(sample)
    chars = len(txt)
    score = min(1.0, chars / (2600.0 if content_kind == "text" else 2200.0))
    return ContentAnalysis(
        content_kind=content_kind,
        text_chars=chars,
        content_score=score,
        script_ratio=0.0,
        blocked=False,
        nextjs=False,
        spa=False,
        html_signals=None,
    )


def extract_html_signals(content: bytes) -> HtmlSignals:
    sample = (content or b"")[:450_000].decode("utf-8", errors="ignore")
    if not sample:
        return HtmlSignals(
            text_chars=0,
            script_ratio=1.0,
            link_density=0.0,
            punctuation_density=0.0,
            visible_text="",
            title_text="",
            heading_text="",
            tag_count=0,
            parser_name="fallback",
        )
    if _SELECTOLAX_AVAILABLE and _SelectolaxHTMLParser is not None:
        try:
            return _extract_html_signals_selectolax(sample)
        except Exception:
            pass
    try:
        return _extract_html_signals_bs4(sample)
    except Exception:
        plain = _collapse_whitespace(sample)
        chars = len(plain)
        return HtmlSignals(
            text_chars=chars,
            script_ratio=0.0,
            link_density=0.0,
            punctuation_density=float(len(_PUNCT_RE.findall(plain)))
            / float(max(1, chars)),
            visible_text=plain,
            title_text="",
            heading_text="",
            tag_count=0,
            parser_name="fallback",
        )


def _extract_html_signals_selectolax(sample: str) -> HtmlSignals:
    parser_factory = cast("type[Any]", _SelectolaxHTMLParser)
    tree = parser_factory(sample)
    title_text = _collapse_whitespace(_node_text(tree.css_first("title")))
    heading_text = _collapse_whitespace(
        " ".join(_node_text(node) for node in tree.css("h1, h2"))
    )
    link_text = _collapse_whitespace(
        " ".join(_node_text(node) for node in tree.css("a"))
    )
    script_nodes = list(tree.css("script"))
    tag_nodes = list(tree.css("*"))
    visible_chunks: list[str] = []
    for node in tree.body.iter() if tree.body is not None else tree.root.iter():
        tag_name = str(getattr(node, "tag", "") or "").lower()
        if tag_name in {"script", "style", "noscript", "svg"}:
            continue
        if tag_name == "-text":
            text = _collapse_whitespace(str(getattr(node, "text", "") or ""))
            if text:
                visible_chunks.append(text)
    visible_text = _collapse_whitespace(" ".join(visible_chunks))
    text_chars = len(visible_text)
    link_density = float(len(link_text)) / float(max(1, text_chars))
    punct_density = float(len(_PUNCT_RE.findall(visible_text))) / float(
        max(1, text_chars)
    )
    return HtmlSignals(
        text_chars=text_chars,
        script_ratio=float(len(script_nodes)) / float(max(1, len(tag_nodes))),
        link_density=link_density,
        punctuation_density=punct_density,
        visible_text=visible_text,
        title_text=title_text,
        heading_text=heading_text,
        tag_count=len(tag_nodes),
        parser_name="selectolax",
    )


def _extract_html_signals_bs4(sample: str) -> HtmlSignals:
    soup = BeautifulSoup(sample, "html.parser")
    script_tags = soup.find_all("script")
    title_tag = soup.find("title")
    title_text = (
        _collapse_whitespace(title_tag.get_text(" ", strip=True))
        if title_tag is not None
        else ""
    )
    heading_text = _collapse_whitespace(
        " ".join(node.get_text(" ", strip=True) for node in soup.find_all(["h1", "h2"]))
    )
    links = " ".join(a.get_text(" ", strip=True) for a in soup.find_all("a"))
    for node in soup.find_all(["script", "style", "noscript", "svg"]):
        node.decompose()
    visible_text = _collapse_whitespace(soup.get_text(" ", strip=True))
    text_chars = len(visible_text)
    all_tags = max(1, len(soup.find_all(True)))
    link_density = float(len(_collapse_whitespace(links))) / float(max(1, text_chars))
    punct_density = float(len(_PUNCT_RE.findall(visible_text))) / float(
        max(1, text_chars)
    )
    return HtmlSignals(
        text_chars=text_chars,
        script_ratio=float(len(script_tags)) / float(all_tags),
        link_density=link_density,
        punctuation_density=punct_density,
        visible_text=visible_text,
        title_text=title_text,
        heading_text=heading_text,
        tag_count=all_tags,
        parser_name="beautifulsoup",
    )


def _node_text(node: Any | None) -> str:
    if node is None:
        return ""
    text_value = getattr(node, "text", None)
    if isinstance(text_value, str):
        return text_value
    method = getattr(node, "text_content", None)
    if callable(method):
        result = method()
        if isinstance(result, str):
            return result
    return ""


def _collapse_whitespace(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()


def estimate_text_quality(
    content: bytes, *, content_kind: str
) -> tuple[int, float, float]:
    content_type: str | None = None
    if content_kind == "html":
        content_type = "text/html"
    elif content_kind == "text":
        content_type = "text/plain"
    elif content_kind == "pdf":
        content_type = "application/pdf"
    analysis = analyze_content(
        content=content,
        content_type=content_type,
        url="https://content.local",
        markers=None,
    )
    return analysis.text_chars, analysis.content_score, analysis.script_ratio


def has_spa_signals(content: bytes) -> bool:
    sample = content[:80_000].decode("utf-8", errors="ignore")
    return bool(_SPA_RE.search(sample))


def has_nextjs_signals(content: bytes) -> bool:
    sample = content[:120_000].decode("utf-8", errors="ignore")
    return bool(_NEXTJS_RE.search(sample))


def blocked_marker_hit(
    content: bytes,
    *,
    markers: tuple[str, ...] | list[str] | None = None,
    html_signals: HtmlSignals | None = None,
) -> bool:
    """Check whether content looks like an actual blocking page."""
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
    visible_text = lowered
    title_text = ""
    heading_text = ""
    if html_signals is not None:
        visible_text = str(html_signals.visible_text or "").lower()
        title_text = str(html_signals.title_text or "").lower()
        heading_text = str(html_signals.heading_text or "").lower()
    elif any(token in lowered for token in ("<html", "<!doctype", "<body", "<head")):
        extracted = extract_html_signals(content[:30_000])
        visible_text = str(extracted.visible_text or "").lower()
        title_text = str(extracted.title_text or "").lower()
        heading_text = str(extracted.heading_text or "").lower()
    total_hits = sum(1 for marker in use_markers if marker in visible_text)
    if total_hits == 0:
        return False
    strong_signals = sum(
        1 for marker in use_markers if marker in title_text or marker in heading_text
    )
    weak_only_signals = sum(
        1
        for marker in use_markers
        if marker in visible_text
        and marker not in title_text
        and marker not in heading_text
    )
    if strong_signals >= 2:
        return True
    if strong_signals >= 1 and total_hits >= 2:
        return True
    if weak_only_signals == total_hits and len(visible_text) > 2000:
        return False
    if total_hits == 1 and strong_signals == 0:
        return False
    if total_hits >= 2:
        return True
    return total_hits >= 1


__all__ = [
    "ContentAnalysis",
    "DEFAULT_USER_AGENT",
    "HtmlSignals",
    "USER_AGENTS",
    "analyze_content",
    "blocked_marker_hit",
    "browser_headers",
    "classify_content_kind",
    "estimate_text_quality",
    "extract_html_signals",
    "get_delay_s",
    "get_random_user_agent",
    "has_nextjs_signals",
    "has_spa_signals",
    "normalize_route_key",
    "parse_retry_after_s",
]
