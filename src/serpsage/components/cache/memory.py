from __future__ import annotations

from typing_extensions import override

import anyio

from serpsage.components.cache.base import CacheBase, CacheConfigBase


class MemoryCacheConfig(CacheConfigBase):
    __setting_family__ = "cache"
    __setting_name__ = "memory"


class MemoryCache(CacheBase[MemoryCacheConfig]):
    store: dict[tuple[str, str], tuple[int, bytes]] = {}
    lock: anyio.Lock = anyio.Lock()

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        now = int(self.clock.now_ms())
        cache_key = (namespace, key)
        async with self.lock:
            item = self.store.get(cache_key)
            if item is None:
                return None
            exp_ms, payload = item
            if int(exp_ms) <= now:
                self.store.pop(cache_key, None)
                return None
            return bytes(payload)

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        now = int(self.clock.now_ms())
        exp_ms = now + int(ttl_s) * 1000
        cache_key = (namespace, key)
        async with self.lock:
            self.store[cache_key] = (exp_ms, bytes(value))
            expired = [k for k, v in self.store.items() if int(v[0]) <= now]
            for k in expired:
                self.store.pop(k, None)


__all__ = ["MemoryCache", "MemoryCacheConfig"]
