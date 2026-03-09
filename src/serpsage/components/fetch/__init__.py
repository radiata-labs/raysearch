from __future__ import annotations

from typing import Any

from serpsage.components.fetch.base import FetcherBase


def build_fetcher(
    *,
    rt: Any,
    rate_limiter: Any | None = None,
    http: Any | None = None,
) -> FetcherBase:
    _ = rate_limiter
    _ = http
    return rt.components.resolve_default("fetch", expected_type=FetcherBase)


__all__ = ["FetcherBase", "build_fetcher"]
