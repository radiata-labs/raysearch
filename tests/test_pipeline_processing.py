from __future__ import annotations

from search_core.config import SearchConfig, SearchContextConfig
from search_core.pipeline import SearchPipeline


def test_empty_query_returns_empty_markdown():
    cfg = SearchConfig(
        default_profile="general",
        profiles={"general": SearchContextConfig()},
    )
    pipeline = SearchPipeline(cfg)
    ctx = pipeline.build_context({"results": []}, "   ")
    assert "用户问题为空" in ctx.markdown


def test_processing_filters_noise_dedupes_and_ranks():
    profile_cfg = SearchContextConfig.model_validate(
        {
            "noise_extensions": ("pdf",),
            "ranking": {
                "strategy": "heuristic",
                "min_relevance_score": 1,
                "min_intent_score": 1,
            },
        }
    )
    cfg = SearchConfig(default_profile="general", profiles={"general": profile_cfg})
    pipeline = SearchPipeline(cfg)

    response = {
        "results": [
            {
                "url": "https://a.example/page",
                "title": "Hello world",
                "snippet": "something",
                "engine": "x",
            },
            # Duplicate (exact)
            {
                "url": "https://a.example/page2",
                "title": "Hello world",
                "snippet": "something",
                "engine": "x",
            },
            # Noise extension
            {
                "url": "https://b.example/file.pdf",
                "title": "Hello pdf",
                "snippet": "hello in pdf",
                "engine": "x",
            },
            # Lower score: token only in snippet
            {
                "url": "https://c.example/page",
                "title": "Other title",
                "snippet": "hello appears here",
                "engine": "x",
            },
            # Irrelevant (no token)
            {
                "url": "https://d.example/page",
                "title": "Nothing",
                "snippet": "nope",
                "engine": "x",
            },
        ]
    }

    ctx = pipeline.build_context(response, "hello", max_results=10)
    assert len(ctx.results) == 2
    assert all(not r.url.endswith(".pdf") for r in ctx.results)
    assert all(0.0 <= r.score <= 1.0 for r in ctx.results)

    # Ranking: title-hit should beat snippet-hit
    assert ctx.results[0].title.lower().startswith("hello")
    assert "网络搜索结果" in ctx.markdown
