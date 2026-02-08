from __future__ import annotations

from dataclasses import dataclass

from search_core.config import SearchConfig, SearchContextConfig
from search_core.pipeline import SearchPipeline
from search_core.web import WebEnricher


@dataclass
class DummyScorer:
    scores: list[float]

    def score(self, texts: list[str], *, query: str, **_kwargs) -> list[tuple[float, str]]:  # noqa: ARG002
        assert len(texts) == len(self.scores)
        return [(float(self.scores[i]), texts[i]) for i in range(len(texts))]


def test_pipeline_min_score_filters_results_and_drops_zero():
    cfg = SearchConfig(
        default_profile="general",
        profiles={"general": SearchContextConfig()},
    )
    # 4 candidates -> keep only >= 0.5 and > 0.0
    dummy = DummyScorer([0.49, 0.5, 0.9, 0.0])
    pipeline = SearchPipeline(cfg, scorer=dummy)  # type: ignore[arg-type]

    response = {
        "results": [
            {"url": "https://a.example/1", "title": "hello a", "snippet": "hello", "engine": "x"},
            {"url": "https://a.example/2", "title": "hello b", "snippet": "hello", "engine": "x"},
            {"url": "https://a.example/3", "title": "hello c", "snippet": "hello", "engine": "x"},
            {"url": "https://a.example/4", "title": "hello d", "snippet": "hello", "engine": "x"},
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
