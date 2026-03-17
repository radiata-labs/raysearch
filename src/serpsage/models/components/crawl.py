from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.models.base import FrozenModel


class CrawlResult(FrozenModel):
    url: str
    status_code: int
    content_type: str | None = None
    content: bytes = b""
    crawl_backend: Literal["curl_cffi", "playwright", "pre_fetched"] = "curl_cffi"
    rendered: bool = False
    content_kind: Literal["html", "pdf", "text", "json", "reddit_json", "doi_json", "binary", "unknown"] = "unknown"
    headers: dict[str, str] = Field(default_factory=dict)
    attempt_chain: list[str] = Field(default_factory=list)


class CrawlAttempt(CrawlResult):
    strategy_used: Literal["curl_cffi", "playwright"] = "curl_cffi"
    content_encoding: str | None = None
    content_length_header: str | None = None
    content_score: float = 0.0
    text_chars: int = 0
    script_ratio: float = 0.0
    blocked: bool = False
    render_reason: str | None = None


__all__ = ["CrawlAttempt", "CrawlResult"]
