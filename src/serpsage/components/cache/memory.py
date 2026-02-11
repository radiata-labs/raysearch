from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.contracts.services import CacheBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class MemoryCache(CacheBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        self._store: dict[tuple[str, str], tuple[int, bytes]] = {}
        self._lock = anyio.Lock()

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        now = int(self.clock.now_ms())
        cache_key = (namespace, key)
        async with self._lock:
            item = self._store.get(cache_key)
            if item is None:
                return None
            exp_ms, payload = item
            if int(exp_ms) <= now:
                self._store.pop(cache_key, None)
                return None
            return bytes(payload)

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        now = int(self.clock.now_ms())
        exp_ms = now + int(ttl_s) * 1000
        cache_key = (namespace, key)
        async with self._lock:
            self._store[cache_key] = (exp_ms, bytes(value))
            expired = [k for k, v in self._store.items() if int(v[0]) <= now]
            for k in expired:
                self._store.pop(k, None)


__all__ = ["MemoryCache"]
