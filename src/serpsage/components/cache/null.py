from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.components.cache.base import CacheBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class NullCache(CacheBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        return None

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        return


__all__ = ["NullCache"]
