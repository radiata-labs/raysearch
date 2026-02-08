from __future__ import annotations

from dataclasses import dataclass

import anyio
import httpx

from search_core.client import AsyncSearxngClient
from search_core.config import SearchConfig, SearchContextConfig
from search_core.searcher import Searcher
from search_core.enrich import WebEnricher


@dataclass
class DummyScorer:
    scores: list[float]

    def score(
        self, texts: list[str], *, query: str, **_kwargs
    ) -> list[tuple[float, str]]:  # noqa: ARG002
        assert len(texts) == len(self.scores)
        return [(float(self.scores[i]), texts[i]) for i in range(len(texts))]


def test_pipeline_min_score_filters_results_and_drops_zero():
    cfg = SearchConfig(
        default_profile="general",
        profiles={"general": SearchContextConfig()},
    )
    # 4 candidates -> keep only >= 0.5 and > 0.0
    dummy = DummyScorer([0.49, 0.5, 0.9, 0.0])
    pipeline = Searcher(cfg, scorer=dummy)  # type: ignore[arg-type]

    response = {
        "results": [
            {
                "url": "https://a.example/1",
                "title": "hello a",
                "snippet": "hello",
                "engine": "x",
            },
            {
                "url": "https://a.example/2",
                "title": "hello b",
                "snippet": "hello",
                "engine": "x",
            },
            {
                "url": "https://a.example/3",
                "title": "hello c",
                "snippet": "hello",
                "engine": "x",
            },
            {
                "url": "https://a.example/4",
                "title": "hello d",
                "snippet": "hello",
                "engine": "x",
            },
        ]
    }

    ctx = pipeline.build_context(response, "hello", "simple", max_results=10)
    assert [round(r.score, 2) for r in ctx.results] == [0.9, 0.5]


def test_web_min_score_filters_chunks_and_drops_zero():
    cfg = SearchConfig(
        default_profile="general",
        profiles={"general": SearchContextConfig()},
    )
    # Make the post-processing stable for this test: disable position-based decay.
    cfg.web_enrichment.select.early_bonus = 1.0
    dummy = DummyScorer([0.4, 0.5, 0.0])
    enricher = WebEnricher(
        cfg.web_enrichment,
        user_agent="ua",
        fetcher=lambda _url: b"",
        scorer=dummy,  # type: ignore[arg-type]
        min_score=float(cfg.score_filter.min_score),
    )

    chunks = ["chunk a", "chunk b", "chunk c"]
    scored = enricher.score_chunks(
        chunks,
        query="x",
        query_tokens=["x"],
        intent_tokens=[],
        domain="example.com",
        context_config=SearchContextConfig(),
    )
    assert scored == [(0.5, "chunk b")]


def test_searxng_async_client_sends_params_and_headers():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)

    cfg = SearchConfig()
    cfg.searxng.base_url = "https://searx.example/search"
    cfg.searxng.search_api_key = "k"
    cfg.searxng.user_agent = "ua-test"

    client = AsyncSearxngClient(cfg, async_client=async_client)

    async def run() -> None:
        out = await client.asearch("q1", params={"lang": "zh"})
        assert out == {"ok": True}

    try:
        anyio.run(run)
    finally:
        anyio.run(async_client.aclose)

    url = str(seen["url"])
    assert "https://searx.example/search" in url
    assert "q=q1" in url
    assert "format=json" in url
    assert "lang=zh" in url
    assert seen["auth"] == "Bearer k"
    assert seen["ua"] == "ua-test"
