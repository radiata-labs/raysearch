from __future__ import annotations

from dataclasses import dataclass

import anyio

from search_core.config import SearchConfig, SearchContextConfig
from search_core.models import PageChunk, PageEnrichment
from search_core.searcher import AsyncSearcher, Searcher


@dataclass
class DummyWebEnricher:
    """Attach deterministic page chunks so page scoring can affect ranking."""

    def enrich_results(self, results, **_kwargs) -> None:  # noqa: ANN001
        # Give result 1 a decent page, result 3 a great page, result 2 none.
        for r in results:
            if str(r.url).endswith("/1"):
                r.page = PageEnrichment(chunks=[PageChunk(text="PAGE1 ok", score=0.9)])
            elif str(r.url).endswith("/3"):
                r.page = PageEnrichment(
                    chunks=[PageChunk(text="PAGE3 excellent", score=0.9)]
                )


class DummyScorer:
    """Return different scores for snippet docs vs page docs; provide normalize_scores()."""

    def score(
        self, texts: list[str], *, query: str, **_kwargs
    ) -> list[tuple[float, str]]:  # noqa: ARG002
        # Heuristic: page docs contain "PAGE".
        if any("PAGE" in (t or "") for t in texts):
            out = []
            for t in texts:
                if "PAGE3" in t:
                    out.append((1.0, t))
                elif "PAGE1" in t:
                    out.append((0.6, t))
                else:
                    out.append((0.0, t))
            return out

        # Snippet-based initial ranking: 1 > 3 > 2
        # Align with the 3 results created in the test.
        return [(0.9, texts[0]), (0.2, texts[1]), (0.7, texts[2])]

    def normalize_scores(self, scores: list[float]) -> list[float]:
        # Mirror rank-based normalization behavior for small n.
        cleaned = [float(s) for s in (scores or [])]
        n = len(cleaned)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        spread = max(cleaned) - min(cleaned)
        if spread < 1e-9:
            return [0.0 for _ in cleaned]

        ordered = sorted(enumerate(cleaned), key=lambda t: t[1], reverse=True)
        out = [0.0 for _ in cleaned]

        i = 0
        while i < n:
            j = i
            while j < n and ordered[j][1] == ordered[i][1]:
                j += 1
            avg_pos = (i + (j - 1)) / 2.0
            percentile = 1.0 - (avg_pos / (n - 1))
            for k in range(i, j):
                out[ordered[k][0]] = float(percentile)
            i = j

        return out


def test_page_score_can_rerank_results():
    cfg = SearchConfig(
        default_profile="general", profiles={"general": SearchContextConfig()}
    )
    # Keep defaults: min_score=0.5 means we keep top 2 after normalization.
    pipeline = Searcher(
        cfg,
        scorer=DummyScorer(),  # type: ignore[arg-type]
        web_enricher=DummyWebEnricher(),  # type: ignore[arg-type]
    )

    response = {
        "results": [
            {
                "url": "https://a.example/1",
                "title": "t1",
                "snippet": "hello s1",
                "engine": "x",
            },
            {
                "url": "https://a.example/2",
                "title": "t2",
                "snippet": "hello s2",
                "engine": "x",
            },
            {
                "url": "https://a.example/3",
                "title": "t3",
                "snippet": "hello s3",
                "engine": "x",
            },
        ]
    }

    ctx = pipeline.build_context(response, "hello", "high", max_results=10)
    # Snippet ranking would be 1 > 3 > 2, but page makes 3 outrank 1.
    assert [r.url for r in ctx.results] == [
        "https://a.example/3",
        "https://a.example/1",
    ]
    assert all(0.0 <= float(r.score) <= 1.0 for r in ctx.results)


@dataclass
class DummyAsyncClient:
    async def asearch(self, query: str, *, params=None) -> dict:  # noqa: ANN001, ARG002
        return {
            "results": [
                {
                    "url": "https://a.example/1",
                    "title": "t1",
                    "snippet": "hello s1",
                    "engine": "x",
                },
                {
                    "url": "https://a.example/2",
                    "title": "t2",
                    "snippet": "hello s2",
                    "engine": "x",
                },
                {
                    "url": "https://a.example/3",
                    "title": "t3",
                    "snippet": "hello s3",
                    "engine": "x",
                },
            ]
        }

    async def aclose(self) -> None:
        return


@dataclass
class DummyAsyncWebEnricher:
    async def aenrich_results(self, results, **_kwargs) -> None:  # noqa: ANN001
        for r in results:
            if str(r.url).endswith("/1"):
                r.page = PageEnrichment(chunks=[PageChunk(text="PAGE1 ok", score=0.9)])
            elif str(r.url).endswith("/3"):
                r.page = PageEnrichment(
                    chunks=[PageChunk(text="PAGE3 excellent", score=0.9)]
                )

    async def aclose(self) -> None:
        return


def test_async_pipeline_can_rerank_results():
    cfg = SearchConfig(
        default_profile="general", profiles={"general": SearchContextConfig()}
    )
    pipeline = AsyncSearcher(
        cfg,
        client=DummyAsyncClient(),  # type: ignore[arg-type]
        scorer=DummyScorer(),  # type: ignore[arg-type]
        web_enricher=DummyAsyncWebEnricher(),  # type: ignore[arg-type]
    )

    async def run() -> list[str]:
        out = await pipeline.asearch_json("hello", "high", max_results=10)
        return [r["url"] for r in out["results"]]  # type: ignore[index]

    urls = anyio.run(run)
    assert urls == ["https://a.example/3", "https://a.example/1"]
