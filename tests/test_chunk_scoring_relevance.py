from __future__ import annotations

from search_core.config import SearchConfig, SearchContextConfig
from search_core.web import WebEnricher


def test_chunk_scoring_prefers_relevant_text():
    profile_cfg = SearchContextConfig.model_validate(
        {
            "ranking": {
                "strategy": "hybrid",
                "min_relevance_score": 1,
                "min_intent_score": 1,
            }
        }
    )
    cfg = SearchConfig(default_profile="general", profiles={"general": profile_cfg})
    enricher = WebEnricher(cfg.web_enrichment, user_agent="ua", fetcher=lambda _: "")

    chunks = [
        "this chunk has nothing",
        "rareterm appears here rareterm",
        "another chunk without it",
    ]

    scored = enricher.score_chunks(
        chunks,
        query="rareterm",
        query_tokens=["rareterm"],
        intent_tokens=[],
        domain="example.com",
        context_config=profile_cfg,
        ranking_config=profile_cfg.ranking,
    )

    best = max(scored, key=lambda item: item[0])[1]
    assert best == chunks[1]
