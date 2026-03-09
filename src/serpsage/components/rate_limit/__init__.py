from __future__ import annotations

from typing import Any

from serpsage.components.rate_limit.base import RateLimiterBase


def build_rate_limiter(*, rt: Any) -> RateLimiterBase:
    return rt.components.resolve_default("rate_limit", expected_type=RateLimiterBase)


__all__ = ["RateLimiterBase", "build_rate_limiter"]
