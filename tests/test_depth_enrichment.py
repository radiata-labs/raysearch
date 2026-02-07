from __future__ import annotations

from search_core.config import SearchConfig, SearchContextConfig
from search_core.pipeline import SearchPipeline


def _make_html_with_tokens(*, length: int = 650, positions: tuple[int, ...] = (0, 200, 400)) -> str:
    buf = ["x"] * length
    for pos in positions:
        if pos + 5 <= length:
            buf[pos : pos + 5] = list("hello")
    text = "".join(buf)
    return f"<html><body>{text}</body></html>"


def test_depth_simple_does_not_crawl():
    profile_cfg = SearchContextConfig.model_validate(
        {
            "ranking": {
                "strategy": "heuristic",
                "min_relevance_score": 1,
                "min_intent_score": 1,
            },
        }
    )
    cfg = SearchConfig(default_profile="general", profiles={"general": profile_cfg})

    def fetcher(_url: str) -> str:  # pragma: no cover
        raise AssertionError("fetcher should not be called for depth=simple")

    pipeline = SearchPipeline(cfg, page_fetcher=fetcher)
    response = {
        "results": [
            {"url": "https://a.example/page", "title": "hello", "snippet": "hello", "engine": "x"},
        ]
    }
    ctx = pipeline.build_context(response, "hello", depth="simple")
    assert ctx.results[0].page_chunks == []


def test_depth_low_enriches_top_results_with_overlapping_chunks():
    profile_cfg = SearchContextConfig.model_validate(
        {
            "ranking": {
                "strategy": "heuristic",
                "min_relevance_score": 1,
                "min_intent_score": 1,
            },
        }
    )
    cfg = SearchConfig(default_profile="general", profiles={"general": profile_cfg})

    html_a = _make_html_with_tokens()
    html_b = _make_html_with_tokens(positions=(0, 300))

    def fetcher(url: str) -> str:
        return html_a if "a.example" in url else html_b

    pipeline = SearchPipeline(cfg, page_fetcher=fetcher)
    response = {
        "results": [
            {"url": "https://a.example/page", "title": "hello title", "snippet": "hello", "engine": "x"},
            {"url": "https://b.example/page", "title": "other", "snippet": "hello", "engine": "x"},
        ]
    }

    ctx = pipeline.build_context(
        response,
        "hello",
        depth="low",
        max_results=2,
        chunk_chars=240,
        chunk_overlap=40,
    )

    assert ctx.json_data["depth"] == "low"
    assert "页面片段" in ctx.markdown

    # low: crawls only the top result, and keeps 2 chunks per page
    assert len(ctx.results[0].page_chunks) == 2
    assert ctx.results[1].page_chunks == []

    a0, a1 = ctx.results[0].page_chunks[:2]
    assert a0[-40:] == a1[:40]

