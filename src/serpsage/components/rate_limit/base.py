from __future__ import annotations

from abc import ABC, abstractmethod

from serpsage.components.base import ComponentBase, ComponentConfigBase


class RateLimiterConfig(ComponentConfigBase):
    global_limit: int = 24
    per_host: int = 4
    politeness_delay_ms: int = 0


class RateLimiterBase(ComponentBase[RateLimiterConfig], ABC):
    @abstractmethod
    async def acquire(self, *, host: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def release(self, *, host: str) -> None:
        raise NotImplementedError


__all__ = ["RateLimiterBase", "RateLimiterConfig"]
