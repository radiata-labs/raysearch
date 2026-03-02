from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from serpsage.components.cache.base import CacheBase

AioredisModule: type[aioredis.Redis] | None = None
try:
    import aioredis as _aioredis

    AioredisModule = _aioredis.Redis
except Exception:  # noqa: BLE001
    AioredisModule = None

if TYPE_CHECKING:
    import aioredis

    from serpsage.core.runtime import Runtime


class RedisCache(CacheBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        if AioredisModule is None:
            raise RuntimeError("aioredis is required for RedisCache")
        self._url = str(self.settings.cache.redis.url)
        self._prefix = str(self.settings.cache.redis.key_prefix)
        self._client: aioredis.Redis | None = None

    def _redis_key(self, *, namespace: str, key: str) -> str:
        if self._prefix:
            return f"{self._prefix}:{namespace}:{key}"
        return f"{namespace}:{key}"

    @override
    async def on_init(self) -> None:
        if self._client is not None:
            return
        if AioredisModule is None:
            raise RuntimeError("aioredis is required for RedisCache")
        self._client = AioredisModule.from_url(
            self._url,
            encoding=None,
            decode_responses=False,
        )

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        if self._client is None:
            raise RuntimeError("redis client is not initialized")
        raw = await self._client.get(self._redis_key(namespace=namespace, key=key))
        if raw is None:
            return None
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        return zlib.decompress(bytes(raw))

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        if self._client is None:
            raise RuntimeError("redis client is not initialized")
        compressed = zlib.compress(value, level=6)
        await self._client.setex(
            self._redis_key(namespace=namespace, key=key),
            int(ttl_s),
            compressed,
        )

    @override
    async def on_close(self) -> None:
        if self._client is None:
            return
        try:
            await cast("Any", self._client).close()
        finally:
            self._client = None


__all__ = ["RedisCache"]
