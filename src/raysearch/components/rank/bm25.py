from __future__ import annotations

import math
from collections import Counter
from typing_extensions import override

from anyio import to_thread

from raysearch.components.base import ComponentConfigBase

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

    BM25_AVAILABLE = True
except Exception:  # noqa: BLE001
    BM25Okapi = None
    BM25_AVAILABLE = False

from raysearch.components.rank.base import RankerBase, RankMode
from raysearch.tokenize import tokenize


class RankBm25Settings(ComponentConfigBase):
    __setting_family__ = "rank"
    __setting_name__ = "bm25"


def _compute_bm25_scores(
    corpus: list[list[str]],
    query_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
) -> list[float]:
    if not corpus or not query_tokens:
        return []

    n_docs = len(corpus)
    doc_lengths = [len(doc) for doc in corpus]
    avgdl = sum(doc_lengths) / n_docs if n_docs > 0 else 1.0
    doc_freqs: dict[str, int] = {}
    for token in query_tokens:
        df = sum(1 for doc in corpus if token in doc)
        doc_freqs[token] = df
    idf: dict[str, float] = {}
    for token, df in doc_freqs.items():
        raw_idf = math.log((n_docs + 1.0) / (df + 0.5))
        idf[token] = max(epsilon, raw_idf)
    scores: list[float] = []
    for doc in corpus:
        score = 0.0
        doc_len = len(doc)
        term_freqs = Counter(doc)
        for token in query_tokens:
            if token not in term_freqs:
                continue
            tf = term_freqs[token]
            token_idf = idf.get(token, epsilon)
            numerator = token_idf * tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * doc_len / avgdl)
            score += numerator / denominator
        scores.append(score)

    return scores


class Bm25Ranker(RankerBase[RankBm25Settings]):
    @override
    async def score_texts(
        self,
        texts: list[str],
        *,
        query: str,
        query_tokens: list[str],
        mode: RankMode = "retrieve",
    ) -> list[float]:
        _ = self._resolve_mode(mode, supported=("retrieve",))
        if not texts:
            return []
        if not BM25_AVAILABLE or BM25Okapi is None:
            return [0.0 for _ in texts]
        if not query_tokens:
            return [0.0 for _ in texts]
        corpus = [tokenize(doc) for doc in texts]
        scores = await to_thread.run_sync(_compute_bm25_scores, corpus, query_tokens)
        return [float(s) for s in scores]


__all__ = ["BM25_AVAILABLE", "Bm25Ranker", "RankBm25Settings"]
