from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.provider.base import SearchProviderBase

if TYPE_CHECKING:
    from serpsage.components.http.base import HttpClientBase
    from serpsage.core.runtime import Runtime


def build_provider(*, rt: Runtime, http: HttpClientBase) -> SearchProviderBase:
    backend = str(rt.settings.provider.backend or "searxng").lower()
    if backend == "searxng":
        from serpsage.components.provider.searxng import SearxngProvider

        return SearxngProvider(rt=rt, http=http)
    if backend == "google":
        from serpsage.components.provider.google import GoogleProvider

        return GoogleProvider(rt=rt, http=http)
    raise ValueError(
        f"unsupported provider backend `{backend}`; expected searxng or google"
    )


__all__ = [
    "SearchProviderBase",
    "build_provider",
]
