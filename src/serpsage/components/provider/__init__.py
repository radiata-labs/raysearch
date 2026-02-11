from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.provider.searxng import SearxngProvider
from serpsage.contracts.services import SearchProviderBase

if TYPE_CHECKING:
    from serpsage.components.fetch.http_client_unit import HttpClientUnit
    from serpsage.core.runtime import Runtime


def build_provider(*, rt: Runtime, http: HttpClientUnit) -> SearchProviderBase:
    backend = str(rt.settings.provider.backend or "searxng").lower()
    if backend == "searxng":
        return SearxngProvider(rt=rt, http=http)
    raise ValueError(f"unsupported provider backend `{backend}`; expected searxng")


__all__ = [
    "SearxngProvider",
    "build_provider",
]
