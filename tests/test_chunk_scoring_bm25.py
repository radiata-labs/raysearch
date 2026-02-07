from __future__ import annotations

import pytest

from search_core.config import SearchConfig, WebEnrichmentConfig
from search_core.web import BM25_AVAILABLE, WebEnricher


@pytest.mark.skipif(not BM25_AVAILABLE, reason="rank_bm25 not installed")
def test_bm25_scoring_prefers_chunk_with_query_term():
    cfg = SearchConfig()
    cfg.web_enrichment = WebEnrichmentConfig.model_validate(
        {
            "scoring": {"strategy": "bm25"},
        }
    )
    enricher = WebEnricher(cfg.web_enrichment, user_agent="ua", fetcher=lambda _: "")

    chunks = [
        "this chunk has nothing",
        "rareterm appears here rareterm",
        "another chunk without it",
    ]
    scores = enricher.score_chunks(
        chunks,
        query="rareterm",
        query_tokens=["rareterm"],
        intent_tokens=[],
    )
    best = max(range(len(scores)), key=lambda i: scores[i])
    assert best == 1

