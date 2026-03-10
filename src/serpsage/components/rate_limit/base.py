from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from serpsage.components.base import ComponentBase, ComponentConfigBase

RateLimiterConfigT = TypeVar("RateLimiterConfigT", bound=ComponentConfigBase)


class RateLimiterBase(
    ComponentBase[RateLimiterConfigT], ABC, Generic[RateLimiterConfigT]
):
    __di_contract__ = True

    @abstractmethod
    async def acquire(self, *, host: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def release(self, *, host: str) -> None:
        raise NotImplementedError


__all__ = ["RateLimiterBase"]
