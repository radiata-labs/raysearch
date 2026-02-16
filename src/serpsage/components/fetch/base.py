from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from serpsage.models.fetch import FetchResult


class FetcherBase(WorkUnit, ABC):
    @abstractmethod
    async def afetch(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> FetchResult:
        raise NotImplementedError
