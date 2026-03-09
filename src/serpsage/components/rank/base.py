from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Generic, Literal
from typing_extensions import TypeVar

from pydantic import Field, model_validator

from serpsage.components.base import ComponentBase, ComponentConfigBase

RankMode = Literal["retrieve", "rerank"]
RankConfigT = TypeVar(
    "RankConfigT",
    bound=ComponentConfigBase,
    default=ComponentConfigBase,
)


class HeuristicRankSettings(ComponentConfigBase):
    early_bonus: float = 1.15
    unique_hit_weight: float = 6.0
    count_weight: float = 1.5
    intent_hit_weight: float = 5.0
    max_count_per_token: int = 5
    temperature: float = 1.0
    min_items_for_sigmoid: int = 5
    flat_spread_eps: float = 1e-9
    z_clip: float = 8.0


class RankBm25Settings(ComponentConfigBase):
    pass


class RankTfidfSettings(ComponentConfigBase):
    pass


class RankCrossEncoderSettings(ComponentConfigBase):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = Field(default=16, ge=1)
    max_length: int = Field(default=512, ge=1)


class RankBlendRerankSettings(ComponentConfigBase):
    retrieve_weight: float = 0.35
    cross_encoder_weight: float = 0.65

    @model_validator(mode="after")
    def _validate_weights(self) -> RankBlendRerankSettings:
        if float(self.retrieve_weight) < 0:
            raise ValueError("rank.blend.rerank.retrieve_weight must be >= 0")
        if float(self.cross_encoder_weight) < 0:
            raise ValueError("rank.blend.rerank.cross_encoder_weight must be >= 0")
        if float(self.retrieve_weight) + float(self.cross_encoder_weight) <= 0:
            raise ValueError("rank.blend.rerank weights must sum to a positive value")
        return self


def _default_rank_blend_providers() -> dict[str, float]:
    return {"heuristic": 0.7, "tfidf": 0.3}


class RankBlendSettings(ComponentConfigBase):
    providers: dict[str, float] = Field(default_factory=_default_rank_blend_providers)
    rerank: RankBlendRerankSettings = Field(default_factory=RankBlendRerankSettings)


class RankerBase(ComponentBase[RankConfigT], ABC, Generic[RankConfigT]):
    def __init__(
        self,
        *,
        rt: object,
        config: RankConfigT,
    ) -> None:
        super().__init__(rt=rt, config=config)
        self._warned_mode_fallbacks: set[tuple[str, str]] = set()

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
    "HeuristicRankSettings",
    "RankBlendRerankSettings",
    "RankBlendSettings",
    "RankBm25Settings",
    "RankCrossEncoderSettings",
    "RankMode",
    "RankTfidfSettings",
    "RankerBase",
]
