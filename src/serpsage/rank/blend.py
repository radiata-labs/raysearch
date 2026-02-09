from __future__ import annotations

from serpsage.contracts.base import Component
from serpsage.contracts.protocols import Clock, Ranker, Telemetry
from serpsage.rank.bm25 import BM25_AVAILABLE, bm25_scores
from serpsage.rank.heuristic import heuristic_scores
from serpsage.rank.normalize import normalize_scores, rank_scales
from serpsage.settings.models import AppSettings


class BlendRanker(Component[None], Ranker):
    def __init__(self, *, settings: AppSettings, telemetry: Telemetry, clock: Clock) -> None:
        super().__init__(settings=settings, telemetry=telemetry, clock=clock)

    def _provider_weights(self) -> dict[str, float]:
        raw = {k: float(v) for k, v in (self.settings.rank.providers or {}).items() if float(v) > 0}
        if raw.get("bm25") and not BM25_AVAILABLE:
            raw.pop("bm25", None)
        if not raw:
            return {"heuristic": 1.0}
        total = sum(raw.values())
        if total <= 0:
            return {"heuristic": 1.0}
        return {k: float(v) / total for k, v in raw.items()}

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

        heur = heuristic_scores(
            texts,
            query=query,
            cfg=self.settings.rank.heuristic,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )

        weights = self._provider_weights()
        heur_w = float(weights.get("heuristic", 0.0))
        bm25_w = float(weights.get("bm25", 0.0))

        blended = [float(s) * heur_w for s in heur]
        if bm25_w > 0:
            bm25_raw = bm25_scores(texts, query=query)
            scaled = rank_scales(bm25_raw)
            max_heur = max(heur) if heur else 0.0
            anchor = float(max_heur) if float(max_heur) > 0 else 1.0
            for i in range(len(blended)):
                blended[i] += float(scaled[i]) * bm25_w * anchor

        return blended

    def normalize(self, *, scores: list[float]) -> list[float]:
        return normalize_scores(scores, self.settings.rank.normalization)


__all__ = ["BlendRanker"]

