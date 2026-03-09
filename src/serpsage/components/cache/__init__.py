from __future__ import annotations

from typing import Any

from serpsage.components.cache.base import CacheBase


def build_cache(*, rt: Any) -> CacheBase:
    return rt.components.resolve_default("cache", expected_type=CacheBase)


__all__ = ["CacheBase", "build_cache"]
