from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from serpsage.core.workunit import WorkUnit
from serpsage.models.components.provider import SearchProviderResponse


class SearchProviderBase(WorkUnit, ABC):
    @abstractmethod
    async def asearch(
        self,
        *,
        query: str,
        page: int = 1,
        language: str = "",
        **kwargs: Any,
    ) -> SearchProviderResponse:
        raise NotImplementedError
