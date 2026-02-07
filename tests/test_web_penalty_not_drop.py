from __future__ import annotations

from search_core.config import SearchConfig, SearchContextConfig
from search_core.web import WebEnricher


def test_web_penalty_not_drop_for_mild_boilerplate_terms():
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

    clean = "rareterm appears here. rareterm appears again."
    mild_tpl = "rareterm appears here. next page. rareterm appears again."

    scored = enricher.score_chunks(
        [clean, mild_tpl],
        query="rareterm",
        query_tokens=["rareterm"],
        intent_tokens=[],
        domain="example.com",
        context_config=profile_cfg,
        ranking_config=profile_cfg.ranking,
    )

    assert scored
    s_clean = next(s for s, c in scored if c == clean)
    s_tpl = next(s for s, c in scored if c == mild_tpl)
    assert s_tpl < s_clean
