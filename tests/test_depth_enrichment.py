from __future__ import annotations

from search_core.config import SearchConfig, SearchContextConfig
from search_core.pipeline import SearchPipeline


def _make_html_sentences(sentences: list[str]) -> str:
    return "<html><body>" + "".join(f"{s}\u3002" for s in sentences) + "</body></html>"


def test_depth_simple_does_not_crawl():
    profile_cfg = SearchContextConfig.model_validate(
        {
            "ranking": {
                "strategy": "heuristic",
                "min_relevance_score": 1,
                "min_intent_score": 1,
            }
        }
    )
    cfg = SearchConfig(default_profile="general", profiles={"general": profile_cfg})

    def fetcher(_url: str) -> str:  # pragma: no cover
        raise AssertionError("fetcher should not be called for depth=simple")

    pipeline = SearchPipeline(cfg, page_fetcher=fetcher)
    response = {
        "results": [
            {
                "url": "https://a.example/page",
                "title": "hello",
                "snippet": "hello",
                "engine": "x",
            }
        ]
    }

    ctx = pipeline.build_context(response, "hello", "simple")
    assert ctx.results[0].page.chunks == []


def test_depth_low_enriches_top_results_with_sentence_overlap():
    profile_cfg = SearchContextConfig.model_validate(
        {
            "ranking": {
                "strategy": "heuristic",
                "min_relevance_score": 1,
                "min_intent_score": 1,
            }
        }
    )
    cfg = SearchConfig(default_profile="general", profiles={"general": profile_cfg})

    html_a = _make_html_sentences(["S1 hello", "S2 hello", "S3 hello"])
    html_b = _make_html_sentences(["B1 hello", "B2 hello", "B3 hello"])

    def fetcher(url: str) -> str:
        return html_a if "a.example" in url else html_b

    pipeline = SearchPipeline(cfg, page_fetcher=fetcher)
    response = {
        "results": [
            {
                "url": "https://a.example/page",
                "title": "hello title",
                "snippet": "hello",
                "engine": "x",
            },
            {
                "url": "https://b.example/page",
                "title": "other",
                "snippet": "hello",
                "engine": "x",
            },
        ]
    }

    ctx = pipeline.build_context(
        response,
        "hello",
        "low",
        max_results=2,
        chunk_target_chars=18,  # force chunking into 2-sentence chunks
        chunk_overlap_sentences=1,
        min_chunk_chars=1,
    )

    assert ctx.json_data["depth"] == "low"
    assert "页面片段" in ctx.markdown

    # low: crawls only the top result, and keeps 2 chunks per page
    assert len(ctx.results[0].page.chunks) == 2
    assert all(0.0 <= c.score <= 1.0 for c in ctx.results[0].page.chunks)
    assert ctx.results[1].page.chunks == []

    a0, a1 = [c.text for c in ctx.results[0].page.chunks[:2]]
    # overlap_sentences=1 => chunk2 starts with chunk1's last sentence
    last_sentence = a0.split("\u3002")[-2].strip() + "\u3002"
    assert a1.strip().startswith(last_sentence)
