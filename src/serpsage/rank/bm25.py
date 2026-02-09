from __future__ import annotations

from typing import Any

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

    BM25_AVAILABLE = True
except Exception:  # noqa: BLE001
    BM25Okapi = None
    BM25_AVAILABLE = False

from serpsage.text.tokenize import tokenize


def bm25_scores(docs: list[str], *, query: str, **_: Any) -> list[float]:
    if not docs:
        return []
    if not BM25_AVAILABLE or BM25Okapi is None:
        return [0.0 for _ in docs]

    q = tokenize(query)
    if not q:
        return [0.0 for _ in docs]

    corpus = [tokenize(doc) for doc in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q)
    return [float(s) for s in scores]


__all__ = ["BM25_AVAILABLE", "bm25_scores"]
