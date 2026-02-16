from __future__ import annotations

from abc import ABC, abstractmethod

from serpsage.core.workunit import WorkUnit


class RankerBase(WorkUnit, ABC):
    @abstractmethod
    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        raise NotImplementedError
