from __future__ import annotations

import time

import anyio

from serpsage import Engine
from serpsage.app.request import (
    FetchChunksRequest,
    FetchContentRequest,
    FetchOverviewRequest,
    FetchRequest,
)
from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import CacheBase, FetcherBase, LLMClientBase, RankerBase
from serpsage.core.runtime import Overrides, Runtime
from serpsage.models.fetch import FetchResult
from serpsage.models.llm import ChatJSONResult, LLMUsage
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
    async def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, object],
        timeout_s: float | None = None,
    ) -> ChatJSONResult:
        _ = model, messages, schema, timeout_s
        return ChatJSONResult(
            data={
                "summary": "Alpha overview summary with enough length for clipping checks.",
                "key_points": [
                    "Alpha key point one with details.",
                    "Alpha key point two with extra context.",
                ],
                "citations": [
                    {
                        "cite_id": "C1",
                        "source_id": "S1",
                        "url": _URL,
                    }
                ],
            },
            usage=LLMUsage(prompt_tokens=12, completion_tokens=24, total_tokens=36),
        )


def _build_runtime(settings: AppSettings) -> Runtime:
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())


def _build_settings() -> AppSettings:
    settings = AppSettings()
    settings.fetch.chunk.min_query_token_hits = 1
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


def test_content_false_still_computes_chunks_and_overview() -> None:
    req = FetchRequest(
        urls=[_URL],
        content=False,
        chunks=FetchChunksRequest(query="alpha", top_k_chunks=2),
        overview=FetchOverviewRequest(query="alpha"),
    )
    resp = anyio.run(_run_fetch, req)
    assert len(resp.results) == 1
    item = resp.results[0]
    assert item.content == ""
    assert item.chunks
    assert item.overview is not None


def test_without_chunks_and_overview_only_returns_content() -> None:
    req = FetchRequest(
        urls=[_URL],
        content=True,
        chunks=None,
        overview=None,
    )
    resp = anyio.run(_run_fetch, req)
    assert len(resp.results) == 1
    item = resp.results[0]
    assert item.content
    assert item.chunks == []
    assert item.overview is None


def test_max_chars_only_limits_output_fields() -> None:
    req = FetchRequest(
        urls=[_URL],
        content=FetchContentRequest(max_chars=120, depth="low"),
        chunks=FetchChunksRequest(query="alpha", top_k_chunks=2, max_chars=40),
        overview=FetchOverviewRequest(query="alpha", max_chars=30),
    )
    resp = anyio.run(_run_fetch, req)

    assert len(resp.results) == 1
    item = resp.results[0]
    assert len(item.content) <= 120
    assert item.chunks
    assert all(len(ch) <= 43 for ch in item.chunks)
    assert item.overview is not None
    assert len(item.overview.summary) <= 33
    assert all(len(text) <= 33 for text in item.overview.key_points)
