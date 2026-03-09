from __future__ import annotations

from typing import Any, cast

from serpsage.components.rate_limit.base import RateLimiterBase


def build_rate_limiter(*, rt: Any) -> RateLimiterBase:
    return cast("RateLimiterBase", rt.services.require(RateLimiterBase))


__all__ = ["RateLimiterBase", "build_rate_limiter"]
