from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.rank.base import RankerBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


def build_ranker(*, rt: Runtime) -> RankerBase:
    backend = str(rt.settings.rank.backend or "blend").lower()
    if backend == "heuristic":
        from serpsage.components.rank.heuristic import HeuristicRanker

        return HeuristicRanker(rt=rt)
    if backend == "bm25":
        from serpsage.components.rank.bm25 import BM25_AVAILABLE, Bm25Ranker

        if not BM25_AVAILABLE:
            raise RuntimeError("rank backend `bm25` is unavailable: install rank_bm25")
        return Bm25Ranker(rt=rt)
    if backend == "blend":
        bm25_w = float(rt.settings.rank.blend.providers.get("bm25", 0.0))
        from serpsage.components.rank.bm25 import BM25_AVAILABLE

        if bm25_w > 0 and not BM25_AVAILABLE:
            raise RuntimeError(
                "rank backend `blend` requires rank_bm25 when blend.providers.bm25 > 0"
            )
        from serpsage.components.rank.blend import BlendRanker

        return BlendRanker(rt=rt)
    raise ValueError(
        f"unsupported rank backend `{backend}`; expected blend|heuristic|bm25"
    )


__all__ = [
    "RankerBase",
    "build_ranker",
]
