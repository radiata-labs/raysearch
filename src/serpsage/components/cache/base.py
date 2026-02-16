from __future__ import annotations

from abc import ABC, abstractmethod

from serpsage.core.workunit import WorkUnit


class CacheBase(WorkUnit, ABC):
    @abstractmethod
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        raise NotImplementedError

    @abstractmethod
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        raise NotImplementedError
