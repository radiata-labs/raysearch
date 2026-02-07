from __future__ import annotations

import html as html_mod
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING
from typing_extensions import override
from urllib.parse import urlparse

import requests

from .utils import TextUtils

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    text: str
    error: str | None = None
    content_type: str | None = None


def fetch_url(
    url: str,
    *,
    timeout: float = 10.0,
    max_bytes: int = 2_000_000,
    user_agent: str = "searxng-bot/0.1",
) -> FetchResult:
    """Fetch a URL as text, with basic safety/size limits."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return FetchResult(text="", error=f"unsupported scheme: {parsed.scheme}")

    headers = {"User-Agent": user_agent}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)  # noqa: S113
    except requests.RequestException as exc:
        return FetchResult(text="", error=str(exc))

    try:
        resp.raise_for_status()
    except requests.RequestException as exc:
        return FetchResult(
            text="", error=str(exc), content_type=resp.headers.get("content-type")
        )

    content_type = resp.headers.get("content-type")
    if content_type:
        ct = content_type.lower()
        if "text/html" not in ct and "text/plain" not in ct:
            return FetchResult(
                text="",
                error=f"unsupported content-type: {content_type}",
                content_type=content_type,
            )

    chunks: list[bytes] = []
    total = 0
    try:
        for part in resp.iter_content(chunk_size=64 * 1024):
            if not part:
                continue
            total += len(part)
            if total > max_bytes:
                return FetchResult(
                    text="",
                    error=f"exceeded max_bytes={max_bytes}",
                    content_type=content_type,
                )
            chunks.append(part)
    finally:
        resp.close()

    data = b"".join(chunks)
    # requests will guess encoding; keep it simple and robust.
    encoding = resp.encoding or "utf-8"
    try:
        text = data.decode(encoding, errors="replace")
    except LookupError:
        text = data.decode("utf-8", errors="replace")
    return FetchResult(text=text, error=None, content_type=content_type)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._buf: list[str] = []
        self._skip_stack: list[str] = []

    @override
    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        lowered = (tag or "").lower()
        if lowered in {"script", "style", "noscript"}:
            self._skip_stack.append(lowered)

    @override
    def handle_endtag(self, tag: str) -> None:
        lowered = (tag or "").lower()
        if self._skip_stack and self._skip_stack[-1] == lowered:
            self._skip_stack.pop()
        if lowered in {"p", "br", "div", "li", "section", "article", "h1", "h2", "h3"}:
            self._buf.append("\n")

    @override
    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if data:
            self._buf.append(data)

    @override
    def handle_entityref(self, name: str) -> None:
        if self._skip_stack:
            return
        self._buf.append(f"&{name};")

    @override
    def handle_charref(self, name: str) -> None:
        if self._skip_stack:
            return
        self._buf.append(f"&#{name};")

    def get_text(self) -> str:
        return "".join(self._buf)


_WS_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Convert HTML to visible text (best-effort, stdlib-only)."""

    if not html:
        return ""
    parser = _VisibleTextParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001
        # If the HTML is malformed, still return something.
        logger.exception("Failed to parse HTML")
    text = html_mod.unescape(parser.get_text())
    return _WS_RE.sub(" ", text).strip()


def chunk_text(
    text: str,
    *,
    chunk_chars: int = 1200,
    overlap: int = 200,
    min_chunk_chars: int = 50,
) -> list[str]:
    """Split text into overlapping chunks (character-based sliding window)."""

    cleaned = TextUtils.clean_whitespace(text or "")
    if not cleaned:
        return []
    if overlap >= chunk_chars:
        raise ValueError("chunk_overlap must be < chunk_chars")

    step = max(1, chunk_chars - overlap)
    out: list[str] = []
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_chars].strip()
        if len(chunk) < min_chunk_chars:
            continue
        out.append(chunk)
        if start + chunk_chars >= len(cleaned):
            break
    return out


def score_chunk(
    chunk: str,
    *,
    query_tokens: Iterable[str],
    intent_tokens: Iterable[str],
) -> float:
    """Score a chunk for query relevance (simple heuristic)."""

    if not chunk:
        return 0.0
    lowered = chunk.lower()

    qt_hits = [t for t in query_tokens if t and t in lowered]
    if not qt_hits:
        return 0.0

    score = 0.0
    score += 3.0 * len(set(qt_hits))
    score += 1.5 * sum(1 for t in intent_tokens if t and t.lower() in lowered)

    # Early hit bonus.
    head = lowered[:200]
    if any(t in head for t in qt_hits):
        score *= 1.2

    # Light length normalization (avoid huge blocks winning only by chance).
    score *= 1.0 + min(0.15, len(chunk) / 8000.0)
    return float(score)


__all__ = ["FetchResult", "fetch_url", "html_to_text", "chunk_text", "score_chunk"]
