from __future__ import annotations

from typing import Any, cast

from serpsage.components.fetch.base import FetcherBase


def build_fetcher(
    *,
    rt: Any,
    rate_limiter: Any | None = None,
    http: Any | None = None,
) -> FetcherBase:
    _ = rate_limiter
    _ = http
    return cast("FetcherBase", rt.services.require(FetcherBase))


__all__ = ["FetcherBase", "build_fetcher"]
