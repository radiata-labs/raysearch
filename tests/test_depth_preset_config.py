from __future__ import annotations

from search_core.config import SearchConfig, SearchContextConfig, WebDepthPreset
from search_core.pipeline import SearchPipeline


def test_depth_preset_controls_pages_and_top_chunks(monkeypatch):
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

    # Override low preset to make behavior deterministic in this test.
    cfg.web_enrichment.depth_presets["low"] = WebDepthPreset(
        pages_ratio=0.9, min_pages=1, max_pages=2, top_chunks_per_page=1
    )

    calls: list[str] = []

    def fetcher(url: str) -> str:
        calls.append(url)
        return "<html><body>hello\u3002hello\u3002hello\u3002</body></html>"

    pipeline = SearchPipeline(cfg, page_fetcher=fetcher)
    response = {
        "results": [
            {
                "url": f"https://a.example/{i}",
                "title": f"hello {i}",
                "snippet": f"hello {i}",
                "engine": "x",
            }
            for i in range(8)
        ]
    }

    ctx = pipeline.build_context(
        response,
        "hello",
        "low",
        max_results=8,
        chunk_target_chars=10,
        chunk_overlap_sentences=0,
        min_chunk_chars=1,
    )

    # max_pages=2 => only 2 pages crawled
    assert len(calls) == 2
    assert len(ctx.results[0].page.chunks) == 1
    assert len(ctx.results[1].page.chunks) == 1
    assert ctx.results[2].page.chunks == []
