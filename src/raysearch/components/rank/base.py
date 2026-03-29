from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Generic, Literal
from typing_extensions import TypeVar

from raysearch.components.base import ComponentBase, ComponentConfigBase

RankMode = Literal["retrieve", "rerank"]
RankConfigT = TypeVar(
    "RankConfigT",
    bound=ComponentConfigBase,
    default=ComponentConfigBase,
)


class RankerBase(ComponentBase[RankConfigT], ABC, Generic[RankConfigT]):
    _warned_mode_fallbacks: set[tuple[str, str]] = set()

    def _resolve_mode(
        self,
        mode: RankMode,
        *,
        supported: tuple[RankMode, ...],
    ) -> RankMode:
        normalized: RankMode = "rerank" if mode == "rerank" else "retrieve"
        if normalized in supported:
            return normalized
        key = (type(self).__name__, normalized)
        if key not in self._warned_mode_fallbacks:
            self._warned_mode_fallbacks.add(key)
            warnings.warn(
                (
                    f"{type(self).__name__} does not support rank mode "
                    f"`{normalized}`; falling back to `retrieve`."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        return "retrieve"

    @abstractmethod
    async def score_texts(
        self,
        texts: list[str],
        *,
        query: str,
        query_tokens: list[str],
        mode: RankMode = "retrieve",
    ) -> list[float]:
        raise NotImplementedError


__all__ = [
    "RankMode",
    "RankerBase",
]
