from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.contracts.services import RateLimiterBase
    from serpsage.core.runtime import Runtime


def build_rate_limiter(*, rt: Runtime) -> RateLimiterBase:
    from serpsage.components.rate_limit.basic import BasicRateLimiter

    return BasicRateLimiter(rt=rt)


__all__ = [
    "build_rate_limiter",
]
