from __future__ import annotations

from typing import Any, cast
from typing_extensions import override

import anyio

from serpsage.components.base import ComponentMeta
from serpsage.components.rank.base import RankBlendSettings, RankerBase, RankMode
from serpsage.components.rank.utils import blend_weighted, rank_scales

_BLEND_META = ComponentMeta(
    version="1.0.0",
    summary="Weighted composite ranker.",
)


class BlendRanker(RankerBase[RankBlendSettings]):
    meta = _BLEND_META

    def __init__(self) -> None:
        super().__init__()
        self._heuristic = _coerce_ranker(
            "heuristic",
            self.components.require_component_optional("rank", "heuristic"),
        )
        self._tfidf = _coerce_ranker(
            "tfidf",
            self.components.require_component_optional("rank", "tfidf"),
        )
        self._bm25 = _coerce_ranker(
            "bm25",
            self.components.require_component_optional("rank", "bm25"),
        )
        self._cross_encoder = _coerce_ranker(
            "cross_encoder",
            self.components.require_component_optional("rank", "cross_encoder"),
        )
        self.bind_deps(
            self._heuristic,
            self._tfidf,
            self._bm25,
            self._cross_encoder,
        )

    def _provider_weights(self) -> dict[str, float]:
        raw = {
            k: float(v)
            for k, v in (self.config.providers or {}).items()
            if float(v) > 0
        }
        if raw.get("bm25") and self._bm25 is None:
            raw.pop("bm25", None)
        if not raw:
            return {"heuristic": 1.0}
        total = sum(raw.values())
        if total <= 0:
            return {"heuristic": 1.0}
        return {k: float(v) / total for k, v in raw.items()}

    def _rerank_weights(self) -> dict[str, float]:
        cfg = self.config.rerank
        raw = {
            "retrieve": float(cfg.retrieve_weight),
            "cross_encoder": float(cfg.cross_encoder_weight),
        }
        if raw["cross_encoder"] > 0 and self._cross_encoder is None:
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
        heur_w = float(weights.get("heuristic", 0.0))
        tfidf_w = float(weights.get("tfidf", 0.0))
        bm25_w = float(weights.get("bm25", 0.0))
        need_bm25 = bm25_w > 0 and self._bm25 is not None
        heur: list[float] | None = None
        tfidf: list[float] | None = None
        bm25_raw: list[float] | None = None

        async def run_heuristic() -> None:
            nonlocal heur
            heuristic = _require_ranker("heuristic", self._heuristic)
            heur = await heuristic.score_texts(
                texts,
                query=query,
                query_tokens=query_tokens,
                mode="retrieve",
            )

        async def run_bm25() -> None:
            nonlocal bm25_raw
            bm25 = _require_ranker("bm25", self._bm25)
            bm25_raw = await bm25.score_texts(
                texts,
                query=query,
                query_tokens=query_tokens,
                mode="retrieve",
            )

        async def run_tfidf() -> None:
            nonlocal tfidf
            tfidf_ranker = _require_ranker("tfidf", self._tfidf)
            tfidf = await tfidf_ranker.score_texts(
                texts,
                query=query,
                query_tokens=query_tokens,
                mode="retrieve",
            )

        async with anyio.create_task_group() as tg:
            if heur_w > 0:
                tg.start_soon(run_heuristic)
            else:
                heur = [0.0] * len(texts)
            if tfidf_w > 0:
                tg.start_soon(run_tfidf)
            else:
                tfidf = [0.0] * len(texts)
            if need_bm25:
                tg.start_soon(run_bm25)
        heur = heur or [0.0] * len(texts)
        tfidf = tfidf or [0.0] * len(texts)
        max_heur = max(heur) if heur else 0.0
        anchor = float(max_heur) if float(max_heur) > 0 else 1.0
        score_map: dict[str, list[float]] = {"heuristic": heur, "tfidf": tfidf}
        transforms: dict[str, object] = {}
        if need_bm25 and bm25_raw is not None:
            score_map["bm25"] = bm25_raw

            def _scale_bm25(
                scores: list[float], *, scale: float = anchor
            ) -> list[float]:
                return [float(x) * scale for x in rank_scales(scores)]

            transforms["bm25"] = _scale_bm25
        return blend_weighted(
            scores=score_map,
            weights=weights,
            transforms=cast("Any", transforms),
        )

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
        if float(rerank_weights.get("cross_encoder", 0.0)) > 0:
            cross_encoder = _require_ranker("cross_encoder", self._cross_encoder)
            cross_encoder_scores = await cross_encoder.score_texts(
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


def _coerce_ranker(
    name: str,
    value: object | None,
) -> RankerBase[Any] | None:
    if value is None:
        return None
    if not isinstance(value, RankerBase):
        raise TypeError(f"rank component `{name}` must implement RankerBase")
    return value


def _require_ranker(
    name: str,
    value: RankerBase[Any] | None,
) -> RankerBase[Any]:
    if value is None:
        raise RuntimeError(f"rank component `{name}` is not enabled")
    return value


__all__ = ["BlendRanker", "blend_weighted"]
