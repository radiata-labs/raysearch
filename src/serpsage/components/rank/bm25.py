from __future__ import annotations

from typing_extensions import override

from anyio import to_thread

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

    BM25_AVAILABLE = True
except Exception:  # noqa: BLE001
    BM25Okapi = None
    BM25_AVAILABLE = False

from serpsage.contracts.services import RankerBase
from serpsage.text.tokenize import tokenize


class Bm25Ranker(RankerBase):
    def __init__(self, *, rt):  # noqa: ANN001
        super().__init__(rt=rt)

    @override
    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str] | None = None,
        intent_tokens: list[str] | None = None,
    ) -> list[float]:
        _ = query_tokens, intent_tokens
        if not texts:
            return []
        if not BM25_AVAILABLE or BM25Okapi is None:
            return [0.0 for _ in texts]

        q = tokenize(query)
        if not q:
            return [0.0 for _ in texts]

        corpus = [tokenize(doc) for doc in texts]
        bm25 = await to_thread.run_sync(BM25Okapi, corpus)
        scores = await to_thread.run_sync(bm25.get_scores, q)
        return [float(s) for s in scores]

    @override
    def normalize(self, *, scores: list[float]) -> list[float]:
        return list(scores or [])


__all__ = ["BM25_AVAILABLE", "Bm25Ranker"]
