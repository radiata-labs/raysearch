from __future__ import annotations

from typing_extensions import override

import anyio

from serpsage.components.base import ComponentMeta
from serpsage.components.cache.base import CacheBase, CacheConfigBase
from serpsage.components.registry import register_component

_MEMORY_CACHE_META = ComponentMeta(
    family="cache",
    name="memory",
    version="1.0.0",
    summary="In-memory TTL cache.",
    provides=("cache.store",),
    config_model=CacheConfigBase,
)


@register_component(meta=_MEMORY_CACHE_META)
class MemoryCache(CacheBase[CacheConfigBase]):
    meta = _MEMORY_CACHE_META

    def __init__(
        self,
        *,
        rt: object,
        config: CacheConfigBase,
    ) -> None:
        super().__init__(rt=rt, config=config)
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
