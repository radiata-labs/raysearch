from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.components.rank.base import RankerBase
from serpsage.components.rank.bm25 import BM25_AVAILABLE, Bm25Ranker
from serpsage.components.rank.heuristic import HeuristicRanker
from serpsage.components.rank.utils import blend_weighted, rank_scales

if TYPE_CHECKING:
    from collections.abc import Callable

    from serpsage.core.runtime import Runtime


class BlendRanker(RankerBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        self._heuristic = HeuristicRanker(rt=rt)
        self._bm25: Bm25Ranker | None = Bm25Ranker(rt=rt) if BM25_AVAILABLE else None
        self.bind_deps(self._heuristic, self._bm25)

    def _provider_weights(self) -> dict[str, float]:
        raw = {
            k: float(v)
            for k, v in (self.settings.rank.blend.providers or {}).items()
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

    @override
    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        if not texts:
            return []
        weights = self._provider_weights()
        heur_w = float(weights.get("heuristic", 0.0))
        bm25_w = float(weights.get("bm25", 0.0))
        need_bm25 = bm25_w > 0 and self._bm25 is not None
        heur: list[float] | None = None
        bm25_raw: list[float] | None = None

        async def run_heuristic() -> None:
            nonlocal heur
            heur = await self._heuristic.score_texts(
                texts=texts,
                query=query,
                query_tokens=query_tokens,
            )

        async def run_bm25() -> None:
            nonlocal bm25_raw
            assert self._bm25 is not None
            bm25_raw = await self._bm25.score_texts(
                texts=texts,
                query=query,
                query_tokens=query_tokens,
            )

        async with anyio.create_task_group() as tg:
            if heur_w > 0:
                tg.start_soon(run_heuristic)
            else:
                heur = [0.0] * len(texts)
            if need_bm25:
                tg.start_soon(run_bm25)
        heur = heur or [0.0] * len(texts)
        max_heur = max(heur) if heur else 0.0
        anchor = float(max_heur) if float(max_heur) > 0 else 1.0
        score_map: dict[str, list[float]] = {"heuristic": heur}
        transforms: dict[str, Callable[[list[float]], list[float]]] = {}
        if need_bm25 and bm25_raw is not None:
            score_map["bm25"] = bm25_raw

            def _scale_bm25(
                scores: list[float], *, scale: float = anchor
            ) -> list[float]:
                return [float(x) * scale for x in rank_scales(scores)]

            transforms["bm25"] = _scale_bm25
        return blend_weighted(scores=score_map, weights=weights, transforms=transforms)


__all__ = ["BlendRanker", "blend_weighted"]
