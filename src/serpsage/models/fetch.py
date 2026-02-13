from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.core.model_base import FrozenModel


class FetchResult(FrozenModel):
    url: str
    status_code: int
    content_type: str | None = None
    content: bytes = b""
    fetch_mode: Literal["httpx", "curl_cffi", "playwright"] = "httpx"
    rendered: bool = False
    content_kind: Literal["html", "pdf", "text", "binary", "unknown"] = "unknown"
    headers: dict[str, str] = Field(default_factory=dict)


class FetchAttempt(FetchResult):
    strategy_used: Literal["httpx", "curl_cffi", "playwright"] = "httpx"
    content_encoding: str | None = None
    content_length_header: str | None = None
    content_score: float = 0.0
    text_chars: int = 0
    blocked: bool = False
    render_reason: str | None = None


__all__ = ["FetchAttempt", "FetchResult"]
