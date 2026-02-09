from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.protocols import Cache

if TYPE_CHECKING:
    from serpsage.app.runtime import CoreRuntime


class NullCache(WorkUnit, Cache):
    def __init__(self, *, rt: CoreRuntime) -> None:
        super().__init__(rt=rt)

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        return None

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        return


__all__ = ["NullCache"]
