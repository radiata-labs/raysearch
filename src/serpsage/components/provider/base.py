from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from collections.abc import Mapping


class SearchProviderBase(WorkUnit, ABC):
    @abstractmethod
    async def asearch(
        self, *, query: str, params: Mapping[str, object] | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError
