from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.rate_limit.base import RateLimiterBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


def build_rate_limiter(*, rt: Runtime) -> RateLimiterBase:
    from serpsage.components.rate_limit.basic import BasicRateLimiter

    return BasicRateLimiter(rt=rt)


__all__ = [
    "RateLimiterBase",
    "build_rate_limiter",
]
