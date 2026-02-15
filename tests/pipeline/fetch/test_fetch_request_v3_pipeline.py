from __future__ import annotations

import time

import anyio

from serpsage import Engine
from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOverviewRequest,
    FetchRequest,
)
from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import CacheBase, FetcherBase, LLMClientBase, RankerBase
from serpsage.core.runtime import Overrides, Runtime
from serpsage.models.fetch import FetchResult
from serpsage.models.llm import ChatResult, LLMUsage
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry

_URL = "https://example.com/article"
_HTML = (
    "<html><head><title>Alpha Doc</title></head><body>"
    "<main><article><h1>Alpha Topic</h1>"
    "<p>Alpha content is repeated to ensure chunk generation and ranking stability. "
    "Alpha systems are evaluated against baseline corpora, and alpha signals are "
    "used for reproducible benchmark reporting across multiple datasets.</p>"
    "<p>Additional alpha notes describe architecture tradeoffs, evaluation setup, "
    "and validation details for overview generation.</p>"
    "</article></main>"
    "<aside><p>Secondary alpha references and related links.</p></aside>"
    "<a href='https://example.com/r1'>r1</a>"
    "<a href='https://example.com/r2'>r2</a>"
    "</body></html>"
).encode("utf-8")


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class _StubCache(CacheBase):
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        _ = namespace, key
        return None

    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        _ = namespace, key, value, ttl_s
        return


class _StubFetcher(FetcherBase):
    async def afetch(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
        allow_render: bool = True,
        rank_index: int = 0,
    ) -> FetchResult:
        _ = timeout_s, allow_render, rank_index
        return FetchResult(
            url=url,
            status_code=200,
            content_type="text/html; charset=utf-8",
            content=_HTML,
            fetch_mode="httpx",
            rendered=False,
            content_kind="html",
            headers={},
            attempt_chain=["httpx"],
            quality_score=0.92,
        )


class _StubRanker(RankerBase):
    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> list[float]:
        _ = query, query_tokens, intent_tokens
        return [1.0 - (i * 0.01) for i in range(len(texts))]


class _StubLLM(LLMClientBase):
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        _ = model, messages, schema, timeout_s
        if schema is None:
            return ChatResult(
                text="Alpha overview summary with extra context.",
                usage=LLMUsage(prompt_tokens=12, completion_tokens=24, total_tokens=36),
            )
        return ChatResult(
            text='{"answer":"alpha"}',
            data={"answer": "alpha"},
            usage=LLMUsage(prompt_tokens=12, completion_tokens=24, total_tokens=36),
        )


def _build_runtime(settings: AppSettings) -> Runtime:
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())


def _build_settings() -> AppSettings:
    settings = AppSettings()
    settings.fetch.abstract.min_query_token_hits = 1
    settings.fetch.render.enabled = False
    settings.fetch.extract.collect_links_default = True
    return settings


async def _run_fetch(req: FetchRequest) -> object:
    settings = _build_settings()
    rt = _build_runtime(settings)
    overrides = Overrides(
        fetcher=_StubFetcher(rt=rt),
        ranker=_StubRanker(rt=rt),
        llm=_StubLLM(rt=rt),
        cache=_StubCache(rt=rt),
    )
    async with Engine.from_settings(settings, overrides=overrides) as engine:
        return await engine.fetch(req)


def test_content_false_still_computes_abstracts_and_overview() -> None:
    req = FetchRequest(
        urls=[_URL],
        content=False,
        abstracts=FetchAbstractsRequest(query="alpha", top_k_abstracts=2),
        overview=FetchOverviewRequest(query="alpha"),
    )
    resp = anyio.run(_run_fetch, req)
    assert len(resp.results) == 1
    item = resp.results[0]
    assert item.content == ""
    assert item.abstracts
    assert isinstance(item.overview, str)
    assert item.overview


def test_without_abstracts_and_overview_only_returns_content() -> None:
    req = FetchRequest(
        urls=[_URL],
        content=True,
        abstracts=None,
        overview=None,
    )
    resp = anyio.run(_run_fetch, req)
    assert len(resp.results) == 1
    item = resp.results[0]
    assert item.content
    assert item.abstracts == []
    assert item.overview is None


def test_max_chars_only_limits_output_fields() -> None:
    req = FetchRequest(
        urls=[_URL],
        content=FetchContentRequest(max_chars=120, depth="low"),
        abstracts=FetchAbstractsRequest(query="alpha", top_k_abstracts=4, max_chars=160),
        overview=FetchOverviewRequest(query="alpha"),
    )
    resp = anyio.run(_run_fetch, req)

    assert len(resp.results) == 1
    item = resp.results[0]
    assert len(item.content) <= 120
    assert item.abstracts
    assert len("\n".join(item.abstracts)) <= 160
    assert item.overview is not None
