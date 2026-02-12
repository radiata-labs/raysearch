from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.contracts.services import SearchProviderBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.http import HttpClient


def build_provider(*, rt: Runtime, http: HttpClient) -> SearchProviderBase:
    backend = str(rt.settings.provider.backend or "searxng").lower()
    if backend == "searxng":
        from serpsage.components.provider.searxng import SearxngProvider

        return SearxngProvider(rt=rt, http=http)
    raise ValueError(f"unsupported provider backend `{backend}`; expected searxng")


__all__ = [
    "build_provider",
]
