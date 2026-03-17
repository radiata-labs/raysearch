from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.models.base import FrozenModel


class CrawlResult(FrozenModel):
    url: str
    status_code: int
    content_type: str | None = None
    content: bytes = b""
    crawl_backend: str = (
        "curl_cffi"  # e.g., "curl_cffi", "playwright", "pre_fetched", "reddit", "doi"
    )
    rendered: bool = False
    content_kind: Literal[
        "html", "pdf", "text", "markdown", "json", "binary", "unknown"
    ] = "unknown"
    headers: dict[str, str] = Field(default_factory=dict)
    attempt_chain: list[str] = Field(default_factory=list)


class CrawlAttempt(CrawlResult):
    """Detailed crawl attempt with quality metrics."""

    content_score: float = 0.0
    text_chars: int = 0
    script_ratio: float = 0.0
    blocked: bool = False


__all__ = ["CrawlAttempt", "CrawlResult"]
