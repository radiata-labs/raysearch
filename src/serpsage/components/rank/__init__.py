from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.rank.blend import BlendRanker
from serpsage.components.rank.bm25 import BM25_AVAILABLE, Bm25Ranker
from serpsage.components.rank.heuristic import HeuristicRanker
from serpsage.contracts.services import RankerBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


def build_ranker(*, rt: Runtime) -> RankerBase:
    backend = str(rt.settings.rank.backend or "blend").lower()
    if backend == "heuristic":
        return HeuristicRanker(rt=rt)
    if backend == "bm25":
        if not BM25_AVAILABLE:
            raise RuntimeError("rank backend `bm25` is unavailable: install rank_bm25")
        return Bm25Ranker(rt=rt)
    if backend == "blend":
        bm25_w = float(rt.settings.rank.blend.providers.get("bm25", 0.0))
        if bm25_w > 0 and not BM25_AVAILABLE:
            raise RuntimeError(
                "rank backend `blend` requires rank_bm25 when blend.providers.bm25 > 0"
            )
        return BlendRanker(rt=rt)
    raise ValueError(
        f"unsupported rank backend `{backend}`; expected blend|heuristic|bm25"
    )


__all__ = [
    "BlendRanker",
    "Bm25Ranker",
    "HeuristicRanker",
    "build_ranker",
]
