from __future__ import annotations

from typing import Any, cast

from serpsage.components.provider.base import SearchProviderBase


def build_provider(*, rt: Any, http: Any | None = None) -> SearchProviderBase:
    _ = http
    return cast("SearchProviderBase", rt.services.require(SearchProviderBase))


__all__ = ["SearchProviderBase", "build_provider"]
