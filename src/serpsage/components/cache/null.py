from __future__ import annotations

from typing_extensions import override

from serpsage.components.base import ComponentMeta
from serpsage.components.cache.base import CacheBase, CacheConfigBase
from serpsage.components.registry import register_component

_NULL_CACHE_META = ComponentMeta(
    family="cache",
    name="null",
    version="1.0.0",
    summary="No-op cache backend.",
    provides=("cache.store",),
    config_model=CacheConfigBase,
)


@register_component(meta=_NULL_CACHE_META)
class NullCache(CacheBase[CacheConfigBase]):
    meta = _NULL_CACHE_META

    def __init__(
        self,
        *,
        rt: object,
        config: CacheConfigBase,
    ) -> None:
        super().__init__(rt=rt, config=config)

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        _ = namespace, key
        return None

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        _ = namespace, key, value, ttl_s


__all__ = ["NullCache"]
