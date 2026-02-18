from __future__ import annotations

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import SearchRequest
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.rank.base import RankerBase
from serpsage.models.pipeline import SearchStepContext
from serpsage.settings.models import AppSettings
from serpsage.steps.search.search import SearchStep


class _DummyProvider(SearchProviderBase):
    def __init__(self, *, rt, by_query: dict[str, list[dict[str, object]]]) -> None:
        super().__init__(rt=rt)
        self.by_query = by_query

    async def asearch(
        self,
        *,
        query: str,
        params: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        del params
        return list(self.by_query.get(query, []))


class _ConstantRanker(RankerBase):
    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        del query, query_tokens
        return [1.0 for _ in texts]


def test_prefetch_applies_additional_weight_and_dedupes_by_url() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    provider = _DummyProvider(
        rt=rt,
        by_query={
            "main query": [
                {"url": "https://a.com/x", "title": "a", "snippet": "x"},
                {"url": "https://b.com/x", "title": "b", "snippet": "x"},
            ],
            "alt query": [
                {"url": "https://a.com/x", "title": "a2", "snippet": "x"},
                {"url": "https://c.com/x", "title": "c", "snippet": "x"},
            ],
        },
    )
    ranker = _ConstantRanker(rt=rt)
    step = SearchStep(rt=rt, provider=provider, ranker=ranker)
    ctx = SearchStepContext(
        settings=settings,
        request=SearchRequest(
            query="main query",
            depth="deep",
            additional_queries=["alt query"],
            max_results=1,
            fetchs={"content": True},
        ),
    )

    out = anyio.run(step.run, ctx)

    assert out.candidate_urls == ["https://a.com/x", "https://b.com/x"]
    assert out.candidate_scores["https://a.com/x"] == 1.0
    assert out.candidate_scores["https://c.com/x"] == 0.8


def test_prefetch_ignores_exclude_domains_when_include_domains_is_set() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    provider = _DummyProvider(
        rt=rt,
        by_query={
            "q": [
                {"url": "https://news.example.com/paper", "title": "a"},
                {"url": "https://other.org/post", "title": "b"},
            ]
        },
    )
    ranker = _ConstantRanker(rt=rt)
    step = SearchStep(rt=rt, provider=provider, ranker=ranker)
    ctx = SearchStepContext(
        settings=settings,
        request=SearchRequest(
            query="q",
            include_domains=["example.com"],
            exclude_domains=["news.example.com"],
            max_results=2,
            fetchs={"content": True},
        ),
    )

    out = anyio.run(step.run, ctx)

    assert out.candidate_urls == ["https://news.example.com/paper"]
