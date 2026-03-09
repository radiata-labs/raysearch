from __future__ import annotations

from typing import Any, cast

from serpsage.components.cache.base import CacheBase


def build_cache(*, rt: Any) -> CacheBase:
    return cast("CacheBase", rt.services.require(CacheBase))


__all__ = ["CacheBase", "build_cache"]
