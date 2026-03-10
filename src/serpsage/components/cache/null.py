from __future__ import annotations

from typing_extensions import override

from serpsage.components.base import ComponentMeta
from serpsage.components.cache.base import CacheBase, CacheConfigBase


class NullCacheConfig(CacheConfigBase):
    __setting_family__ = "cache"
    __setting_name__ = "null"


_NULL_CACHE_META = ComponentMeta(
    version="1.0.0",
    summary="No-op cache backend.",
)


class NullCache(CacheBase[NullCacheConfig]):
    meta = _NULL_CACHE_META

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        _ = namespace
        _ = key
        return None

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        _ = namespace
        _ = key
        _ = value
        _ = ttl_s


__all__ = ["NullCache", "NullCacheConfig"]
