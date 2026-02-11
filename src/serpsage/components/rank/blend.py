from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.components.rank.bm25 import BM25_AVAILABLE, Bm25Ranker
from serpsage.components.rank.heuristic import HeuristicRanker
from serpsage.components.rank.utils import normalize_scores, rank_scales
from serpsage.contracts.services import RankerBase

if TYPE_CHECKING:
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
    def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str] | None = None,
        intent_tokens: list[str] | None = None,
    ) -> list[float]:
        if not texts:
            return []

        heur = self._heuristic.score_texts(
            texts=texts,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )

        weights = self._provider_weights()
        heur_w = float(weights.get("heuristic", 0.0))
        bm25_w = float(weights.get("bm25", 0.0))

        blended = [float(s) * heur_w for s in heur]
        if bm25_w > 0 and self._bm25 is not None:
            bm25_raw = self._bm25.score_texts(texts=texts, query=query)
            scaled = rank_scales(bm25_raw)
            max_heur = max(heur) if heur else 0.0
            anchor = float(max_heur) if float(max_heur) > 0 else 1.0
            for i in range(len(blended)):
                blended[i] += float(scaled[i]) * bm25_w * anchor

        return blended

    @override
    def normalize(self, *, scores: list[float]) -> list[float]:
        return normalize_scores(scores, self.settings.rank.normalization)


__all__ = ["BlendRanker"]
