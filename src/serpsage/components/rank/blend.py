from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import anyio
from pydantic import Field, model_validator

from serpsage.components.base import ComponentConfigBase
from serpsage.components.rank.base import (
    RankerBase,
    RankMode,
)
from serpsage.components.rank.bm25 import Bm25Ranker
from serpsage.components.rank.cross_encoder import CrossEncoderRanker
from serpsage.components.rank.tfidf import TfidfRanker
from serpsage.components.rank.utils import blend_weighted, rank_scales
from serpsage.dependencies import Depends

if TYPE_CHECKING:
    from collections.abc import Callable


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
    return {"tfidf": 0.5, "bm25": 0.5}


class RankBlendSettings(ComponentConfigBase):
    __setting_family__ = "rank"
    __setting_name__ = "blend"

    providers: dict[str, float] = Field(default_factory=_default_rank_blend_providers)
    rerank: RankBlendRerankSettings = Field(default_factory=RankBlendRerankSettings)


class BlendRanker(RankerBase[RankBlendSettings]):
    tfidf: TfidfRanker = Depends()
    bm25: Bm25Ranker = Depends()
    cross_encoder: CrossEncoderRanker = Depends()

    def _provider_weights(self) -> dict[str, float]:
        raw = {
            k: float(v)
            for k, v in (self.config.providers or {}).items()
            if float(v) > 0
        }
        if raw.get("bm25") and not isinstance(self.bm25, Bm25Ranker):
            raw.pop("bm25", None)
        if raw.get("tfidf") and not isinstance(self.tfidf, TfidfRanker):
            raw.pop("tfidf", None)
        if not raw:
            return {"tfidf": 1.0}
        total = sum(raw.values())
        if total <= 0:
            return {"tfidf": 1.0}
        return {k: float(v) / total for k, v in raw.items()}

    def _rerank_weights(self) -> dict[str, float]:
        cfg = self.config.rerank
        raw = {
            "retrieve": float(cfg.retrieve_weight),
            "cross_encoder": float(cfg.cross_encoder_weight),
        }
        if raw["cross_encoder"] > 0 and not isinstance(
            self.cross_encoder, CrossEncoderRanker
        ):
            raw["cross_encoder"] = 0.0
        filtered = {name: value for name, value in raw.items() if value > 0}
        if not filtered:
            return {"retrieve": 1.0}
        total = sum(filtered.values())
        if total <= 0:
            return {"retrieve": 1.0}
        return {name: value / total for name, value in filtered.items()}

    async def _score_retrieve(
        self,
        texts: list[str],
        *,
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        if not texts:
            return []
        weights = self._provider_weights()
        tfidf_w = float(weights.get("tfidf", 0.0))
        bm25_w = float(weights.get("bm25", 0.0))
        need_bm25 = bm25_w > 0 and isinstance(self.bm25, Bm25Ranker)
        tfidf: list[float] | None = None
        bm25_raw: list[float] | None = None

        async def run_bm25() -> None:
            nonlocal bm25_raw
            bm25_raw = await self.bm25.score_texts(
                texts,
                query=query,
                query_tokens=query_tokens,
                mode="retrieve",
            )

        async def run_tfidf() -> None:
            nonlocal tfidf
            tfidf = await self.tfidf.score_texts(
                texts,
                query=query,
                query_tokens=query_tokens,
                mode="retrieve",
            )

        async with anyio.create_task_group() as tg:
            if tfidf_w > 0:
                tg.start_soon(run_tfidf)
            else:
                tfidf = [0.0] * len(texts)
            if need_bm25:
                tg.start_soon(run_bm25)
        tfidf = tfidf or [0.0] * len(texts)
        max_tfidf = max(tfidf) if tfidf else 0.0
        anchor = float(max_tfidf) if float(max_tfidf) > 0 else 1.0
        score_map: dict[str, list[float]] = {"tfidf": tfidf}
        transforms: dict[str, Callable[[list[float]], list[float]]] = {}
        if need_bm25 and bm25_raw is not None:
            score_map["bm25"] = bm25_raw

            def _scale_bm25(
                scores: list[float], *, scale: float = anchor
            ) -> list[float]:
                return [float(x) * scale for x in rank_scales(scores)]

            transforms["bm25"] = _scale_bm25
        return blend_weighted(scores=score_map, weights=weights, transforms=transforms)

    @override
    async def score_texts(
        self,
        texts: list[str],
        *,
        query: str,
        query_tokens: list[str],
        mode: RankMode = "retrieve",
    ) -> list[float]:
        resolved_mode = self._resolve_mode(mode, supported=("retrieve", "rerank"))
        retrieve_scores = await self._score_retrieve(
            texts,
            query=query,
            query_tokens=query_tokens,
        )
        if resolved_mode == "retrieve":
            return retrieve_scores
        rerank_weights = self._rerank_weights()
        cross_encoder_scores = [0.0 for _ in texts]
        if float(rerank_weights.get("cross_encoder", 0.0)) > 0 and self.cross_encoder:
            cross_encoder_scores = await self.cross_encoder.score_texts(
                texts,
                query=query,
                query_tokens=query_tokens,
                mode="rerank",
            )
        return blend_weighted(
            scores={
                "retrieve": retrieve_scores,
                "cross_encoder": cross_encoder_scores,
            },
            weights=rerank_weights,
        )


__all__ = [
    "BlendRanker",
    "RankBlendRerankSettings",
    "RankBlendSettings",
    "blend_weighted",
]
