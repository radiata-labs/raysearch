from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from pydantic import field_validator

from serpsage.components.base import ComponentMeta
from serpsage.components.cache.base import CacheBase, CacheConfigBase
from serpsage.components.registry import register_component

AioredisModule: Any | None = None
try:
    import aioredis as _aioredis

    AioredisModule = _aioredis.Redis
except Exception:  # noqa: BLE001
    AioredisModule = None
if TYPE_CHECKING:
    import aioredis


class RedisCacheConfig(CacheConfigBase):
    url: str = "redis://127.0.0.1:6379/0"
    key_prefix: str = "serpsage"

    @field_validator("url", "key_prefix")
    @classmethod
    def _validate_strings(cls, value: str) -> str:
        return str(value or "").strip()


_REDIS_CACHE_META = ComponentMeta(
    family="cache",
    name="redis",
    version="1.0.0",
    summary="Redis-backed cache.",
    provides=("cache.store",),
    config_model=RedisCacheConfig,
)


@register_component(meta=_REDIS_CACHE_META)
class RedisCache(CacheBase[RedisCacheConfig]):
    meta = _REDIS_CACHE_META

    def __init__(self) -> None:
        if AioredisModule is None:
            raise RuntimeError("aioredis is required for RedisCache")
        self._url = str(self.config.url)
        self._prefix = str(self.config.key_prefix)
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


__all__ = ["RedisCache", "RedisCacheConfig"]
