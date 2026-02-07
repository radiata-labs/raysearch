from __future__ import annotations

from search_core.config import SearchConfig, SearchContextConfig
from search_core.web import WebEnricher


def test_template_chunks_do_not_beat_body_text():
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
    enricher = WebEnricher(cfg.web_enrichment, user_agent="ua", fetcher=lambda _: "")

    template_chunk = (
        "2026\u5e742\u6708 2026\u5e741\u6708 2025\u5e7412\u6708 "
        "2025\u5e7411\u6708 \u30a2\u30fc\u30ab\u30a4\u30d6"
    )
    body_chunk = (
        "\u521d\u97f3\u30df\u30af\u306e\u65b0\u4f5cMV\u304c\u516c\u958b\u3055\u308c\u305f"
    )

    scored = enricher.score_chunks(
        [template_chunk, body_chunk],
        query="\u521d\u97f3\u30df\u30af MV",
        query_tokens=["\u521d\u97f3\u30df\u30af", "mv"],
        intent_tokens=[],
        domain="blog.example",
        context_config=profile_cfg,
        ranking_config=profile_cfg.ranking,
    )

    assert scored
    best_chunk = max(scored, key=lambda t: t[0])[1]
    assert "\u521d\u97f3\u30df\u30af" in best_chunk

    tpl_scores = [s for s, c in scored if "\u30a2\u30fc\u30ab\u30a4\u30d6" in c]
    body_scores = [s for s, c in scored if "\u521d\u97f3\u30df\u30af" in c]
    assert body_scores
    if tpl_scores:
        assert max(tpl_scores) < max(body_scores)
