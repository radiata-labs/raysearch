from __future__ import annotations

from typing import Any

from serpsage.components.provider.base import SearchProviderBase


def build_provider(*, rt: Any, http: Any | None = None) -> SearchProviderBase:
    _ = http
    return rt.components.resolve_default("provider", expected_type=SearchProviderBase)


__all__ = ["SearchProviderBase", "build_provider"]
