from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic
from typing_extensions import TypeVar

from raysearch.components.base import ComponentBase, ComponentConfigBase


class CacheConfigBase(ComponentConfigBase):
    fetch_ttl_s: int = 86_400


CacheConfigT = TypeVar(
    "CacheConfigT",
    bound=CacheConfigBase,
    default=CacheConfigBase,
)


class CacheBase(ComponentBase[CacheConfigT], ABC, Generic[CacheConfigT]):
    @abstractmethod
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        raise NotImplementedError

    @abstractmethod
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        raise NotImplementedError


__all__ = ["CacheBase", "CacheConfigBase"]
