from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.rank.base import RankerBase, RankMode

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


def build_ranker(*, rt: Runtime) -> RankerBase:
    backend = str(rt.settings.rank.backend or "blend").lower()
    if backend == "heuristic":
        from serpsage.components.rank.heuristic import HeuristicRanker

        return HeuristicRanker(rt=rt)
    if backend == "tfidf":
        from serpsage.components.rank.tfidf import TfidfRanker

        return TfidfRanker(rt=rt)
    if backend == "cross_encoder":
        from serpsage.components.rank.cross_encoder import (
            CROSS_ENCODER_AVAILABLE,
            CrossEncoderRanker,
        )

        if not CROSS_ENCODER_AVAILABLE:
            raise RuntimeError(
                "rank backend `cross_encoder` is unavailable: install sentence-transformers"
            )
        return CrossEncoderRanker(rt=rt)
    if backend == "bm25":
        from serpsage.components.rank.bm25 import BM25_AVAILABLE, Bm25Ranker

        if not BM25_AVAILABLE:
            raise RuntimeError("rank backend `bm25` is unavailable: install rank_bm25")
        return Bm25Ranker(rt=rt)
    if backend == "blend":
        bm25_w = float(rt.settings.rank.blend.providers.get("bm25", 0.0))
        cross_encoder_w = float(rt.settings.rank.blend.rerank.cross_encoder_weight)
        from serpsage.components.rank.bm25 import BM25_AVAILABLE
        from serpsage.components.rank.cross_encoder import CROSS_ENCODER_AVAILABLE

        if bm25_w > 0 and not BM25_AVAILABLE:
            raise RuntimeError(
                "rank backend `blend` requires rank_bm25 when blend.providers.bm25 > 0"
            )
        if cross_encoder_w > 0 and not CROSS_ENCODER_AVAILABLE:
            raise RuntimeError(
                "rank backend `blend` requires sentence-transformers when "
                "blend.rerank.cross_encoder_weight > 0"
            )
        from serpsage.components.rank.blend import BlendRanker

        return BlendRanker(rt=rt)
    raise ValueError(
        "unsupported rank backend "
        f"`{backend}`; expected blend|heuristic|tfidf|bm25|cross_encoder"
    )


__all__ = [
    "RankMode",
    "RankerBase",
    "build_ranker",
]
