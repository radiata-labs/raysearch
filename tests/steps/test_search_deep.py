from __future__ import annotations

import time
from typing import Any

import pytest

from serpsage.app.request import FetchRequestBase, SearchRequest
from serpsage.app.response import FetchResultItem
from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.rank.base import RankerBase
from serpsage.core.runtime import Runtime
from serpsage.models.llm import ChatResult
from serpsage.models.pipeline import (
    SearchFetchedCandidate,
    SearchQueryJob,
    SearchStepContext,
)
from serpsage.settings.models import AppSettings
from serpsage.steps.base import RunnerBase
from serpsage.steps.search.expand import SearchExpandStep
from serpsage.steps.search.fetch import SearchFetchStep
from serpsage.steps.search.finalize import SearchFinalizeStep
from serpsage.steps.search.rank import SearchRankStep
from serpsage.steps.search.search import SearchStep
from serpsage.telemetry.base import ClockBase
from serpsage.telemetry.trace import NoopTelemetry
from serpsage.utils import clean_whitespace


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _TestClock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _build_runtime() -> Runtime:
    settings = AppSettings()
    settings.search.deep.rule_max_queries = 4
    settings.search.deep.llm_max_queries = 2
    settings.search.deep.max_expanded_queries = 6
    settings.search.deep.prefetch_multiplier = 3.0
    settings.search.deep.prefetch_max_urls = 12
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_TestClock())


def _build_search_request(
    *,
    query: str,
    mode: str = "deep",
    max_results: int = 3,
    additional_queries: list[str] | None = None,
) -> SearchRequest:
    return SearchRequest(
        query=query,
        mode=mode,
        max_results=max_results,
        additional_queries=additional_queries,
        fetchs=FetchRequestBase(content=True, abstracts=False, overview=False),
    )


def _build_search_context(
    *,
    rt: Runtime,
    query: str,
    mode: str = "deep",
    max_results: int = 3,
    additional_queries: list[str] | None = None,
) -> SearchStepContext:
    req = _build_search_request(
        query=query,
        mode=mode,
        max_results=max_results,
        additional_queries=additional_queries,
    )
    return SearchStepContext(settings=rt.settings, request=req, request_id="search-req")


class _FakeLLM(LLMClientBase):
    def __init__(self, *, rt: Runtime, outputs: list[ChatResult | Exception]) -> None:
        super().__init__(rt=rt)
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "schema": schema,
                "timeout_s": timeout_s,
            }
        )
        if not self._outputs:
            return ChatResult(data={"queries": []}, text='{"queries":[]}')
        item = self._outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeProvider(SearchProviderBase):
    def __init__(self, *, rt: Runtime, responses: dict[str, list[dict[str, Any]]]) -> None:
        super().__init__(rt=rt)
        self._responses = {
            clean_whitespace(key): [dict(item) for item in value]
            for key, value in responses.items()
        }
        self.calls: list[str] = []

    async def asearch(
        self, *, query: str, params: dict[str, object] | None = None
    ) -> list[dict[str, Any]]:
        _ = params
        normalized = clean_whitespace(query)
        self.calls.append(normalized)
        return [dict(item) for item in self._responses.get(normalized, [])]


class _TokenOverlapRanker(RankerBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        _ = query
        tokens = [clean_whitespace(item).casefold() for item in list(query_tokens or [])]
        token_set = {item for item in tokens if item}
        if not token_set:
            return [0.0 for _ in texts]
        out: list[float] = []
        for text in texts:
            normalized = clean_whitespace(str(text or "")).casefold()
            hits = sum(1 for token in token_set if token in normalized)
            out.append(float(hits / len(token_set)))
        return out


def _make_result(*, url: str, title: str, content: str, abstracts: list[str]) -> FetchResultItem:
    return FetchResultItem(
        url=url,
        title=title,
        content=content,
        abstracts=list(abstracts),
        abstract_scores=[0.2 for _ in abstracts],
        subpages=[],
        overview="",
    )


@pytest.mark.anyio
async def test_search_expand_merges_manual_rule_and_llm_queries() -> None:
    rt = _build_runtime()
    llm = _FakeLLM(
        rt=rt,
        outputs=[ChatResult(data={"queries": ["python asyncio event loop internals"]})],
    )
    step = SearchExpandStep(rt=rt, llm=llm)
    ctx = _build_search_context(
        rt=rt,
        query="python async tutorial",
        mode="deep",
        additional_queries=["manual async query"],
    )

    out = await step.run(ctx)

    assert out.deep.aborted is False
    assert out.deep.query_jobs
    assert out.deep.query_jobs[0].source == "primary"
    assert any(item.source == "manual" for item in out.deep.query_jobs)
    assert any(item.source == "rule" for item in out.deep.query_jobs)
    assert any(item.source == "llm" for item in out.deep.query_jobs)
    assert len(out.deep.query_jobs) <= 1 + rt.settings.search.deep.max_expanded_queries


@pytest.mark.anyio
async def test_search_expand_llm_failure_aborts_deep_search() -> None:
    rt = _build_runtime()
    llm = _FakeLLM(rt=rt, outputs=[RuntimeError("llm unavailable")])
    step = SearchExpandStep(rt=rt, llm=llm)
    ctx = _build_search_context(rt=rt, query="latest alpha release", mode="deep")

    out = await step.run(ctx)

    assert out.deep.aborted is True
    assert out.deep.query_jobs == []
    assert out.output.results == []
    assert any(err.code == "search_query_expansion_failed" for err in out.errors)


@pytest.mark.anyio
async def test_search_step_uses_expanded_jobs_and_collects_snippet_context() -> None:
    rt = _build_runtime()
    provider = _FakeProvider(
        rt=rt,
        responses={
            "alpha query": [
                {
                    "url": "https://a.example.com",
                    "title": "A",
                    "snippet": "alpha coverage primary",
                }
            ],
            "manual alpha": [
                {
                    "url": "https://a.example.com",
                    "title": "A2",
                    "snippet": "alpha coverage manual",
                },
                {
                    "url": "https://b.example.com",
                    "title": "B",
                    "snippet": "beta only",
                },
            ],
        },
    )
    ranker = _TokenOverlapRanker(rt=rt)
    step = SearchStep(rt=rt, provider=provider, ranker=ranker)
    ctx = _build_search_context(rt=rt, query="alpha query", mode="deep", max_results=2)
    ctx.deep.query_jobs = [
        SearchQueryJob(query="alpha query", weight=1.0, source="primary"),
        SearchQueryJob(query="manual alpha", weight=0.8, source="manual"),
    ]

    out = await step.run(ctx)

    assert provider.calls == ["alpha query", "manual alpha"]
    assert out.prefetch.urls
    assert out.prefetch.urls[0] == "https://a.example.com"
    assert out.deep.query_hit_stats.get("https://a.example.com") == 2
    assert "https://a.example.com" in out.deep.snippet_context
    assert len(out.deep.snippet_context["https://a.example.com"]) <= 3


@pytest.mark.anyio
async def test_search_step_fast_mode_keeps_single_query_behavior() -> None:
    rt = _build_runtime()
    provider = _FakeProvider(
        rt=rt,
        responses={
            "alpha query": [
                {
                    "url": "https://auto.example.com",
                    "title": "auto",
                    "snippet": "alpha query evidence",
                }
            ]
        },
    )
    step = SearchStep(rt=rt, provider=provider, ranker=_TokenOverlapRanker(rt=rt))
    ctx = _build_search_context(rt=rt, query="alpha query", mode="fast", max_results=1)

    out = await step.run(ctx)

    assert provider.calls == ["alpha query"]
    assert out.prefetch.urls == ["https://auto.example.com"]


@pytest.mark.anyio
async def test_search_fetch_and_finalize_noop_when_deep_aborted() -> None:
    rt = _build_runtime()
    fetch_runner = RunnerBase(rt=rt, steps=[], kind="fetch")
    fetch_step = SearchFetchStep(rt=rt, fetch_runner=fetch_runner)
    finalize_step = SearchFinalizeStep(rt=rt)
    ctx = _build_search_context(rt=rt, query="alpha beta", mode="deep")
    ctx.deep.aborted = True
    ctx.prefetch.urls = ["https://x.example.com"]
    ctx.output.results = [
        _make_result(
            url="https://x.example.com",
            title="x",
            content="x content",
            abstracts=["x abstract"],
        )
    ]

    out_after_fetch = await fetch_step.run(ctx)
    out_after_finalize = await finalize_step.run(out_after_fetch)

    assert out_after_fetch.fetch.candidates == []
    assert out_after_fetch.output.results == []
    assert out_after_finalize.output.results == []


@pytest.mark.anyio
async def test_search_finalize_deep_composite_prefers_context_score() -> None:
    rt = _build_runtime()
    rt.settings.search.deep = rt.settings.search.deep.model_copy(
        update={
            "final_page_weight": 0.1,
            "final_context_weight": 0.8,
            "final_prefetch_weight": 0.1,
        }
    )

    ranker = _TokenOverlapRanker(rt=rt)
    rank_step = SearchRankStep(rt=rt, ranker=ranker)
    step = SearchFinalizeStep(rt=rt)
    ctx = _build_search_context(rt=rt, query="alpha beta", mode="deep", max_results=2)
    ctx.fetch.candidates = [
        SearchFetchedCandidate(
            result=_make_result(
                url="https://one.example.com",
                title="one",
                content="neutral body",
                abstracts=[],
            ),
            main_md_for_abstract="neutral content",
        ),
        SearchFetchedCandidate(
            result=_make_result(
                url="https://two.example.com",
                title="two",
                content="neutral body",
                abstracts=[],
            ),
            main_md_for_abstract="neutral content",
        ),
    ]
    ctx.output.results = [item.result for item in ctx.fetch.candidates]
    ctx.deep.snippet_context = {
        "https://one.example.com": [
            {
                "snippet": "random unrelated text",
                "source_query": "q1",
                "source_type": "primary",
                "score": 0.1,
                "order": 0,
            }
        ],
        "https://two.example.com": [
            {
                "snippet": "alpha beta in-depth details",
                "source_query": "q2",
                "source_type": "manual",
                "score": 0.2,
                "order": 0,
            }
        ],
    }
    ctx.prefetch.scores = {
        "https://one.example.com": 0.1,
        "https://two.example.com": 0.1,
    }

    ranked = await rank_step.run(ctx)
    out = await step.run(ranked)

    assert out.output.results[0].url == "https://two.example.com"
    assert out.deep.context_scores["https://two.example.com"] > out.deep.context_scores[
        "https://one.example.com"
    ]
