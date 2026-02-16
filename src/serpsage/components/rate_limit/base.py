from __future__ import annotations

from abc import ABC, abstractmethod

from serpsage.core.workunit import WorkUnit


class RateLimiterBase(WorkUnit, ABC):
    @abstractmethod
    async def acquire(self, *, host: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def release(self, *, host: str) -> None:
        raise NotImplementedError
