from __future__ import annotations

from typing_extensions import override

from anyio import to_thread
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from serpsage.components.base import ComponentConfigBase
from serpsage.components.rank.base import RankerBase, RankMode
from serpsage.tokenize import tokenize
from serpsage.utils import normalize_text


class RankTfidfSettings(ComponentConfigBase):
    __setting_family__ = "rank"
    __setting_name__ = "tfidf"


def _build_query_terms(query: str, query_tokens: list[str]) -> list[str]:
    candidates = query_tokens or tokenize(query)
    seen: set[str] = set()
    terms: list[str] = []
    for token in candidates:
        normalized = normalize_text(token)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def _score_texts_sync(
    texts: list[str], query: str, query_tokens: list[str]
) -> list[float]:
    if not texts:
        return []
    query_terms = _build_query_terms(query, query_tokens)
    if not query_terms:
        return [0.0 for _ in texts]
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,  # type: ignore
        vocabulary=query_terms,
        lowercase=False,
        norm="l2",
        smooth_idf=True,
        sublinear_tf=True,
    )
    doc_matrix = vectorizer.fit_transform(texts)
    query_matrix = vectorizer.transform([" ".join(query_terms)])
    scores = linear_kernel(query_matrix, doc_matrix).ravel().tolist()
    return [float(min(1.0, max(0.0, score))) for score in scores]


class TfidfRanker(RankerBase[RankTfidfSettings]):
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
        return await to_thread.run_sync(_score_texts_sync, texts, query, query_tokens)


__all__ = ["RankTfidfSettings", "TfidfRanker"]
