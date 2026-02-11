from __future__ import annotations

from serpsage.core.model_base import FrozenModel


class FetchResult(FrozenModel):
    url: str
    status_code: int
    content_type: str | None = None
    content: bytes = b""


class FetchAttempt(FetchResult):
    strategy_used: str = ""
    content_encoding: str | None = None
    content_length_header: str | None = None


__all__ = ["FetchAttempt", "FetchResult"]
