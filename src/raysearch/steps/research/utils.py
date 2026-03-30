"""Research step utility functions.

This module contains pure utility functions used across research steps.
It is one of the two allowed pure-function files (along with prompt.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raysearch.models.steps.research import (
    ResearchSource,
)
from raysearch.settings.models import AppSettings
from raysearch.tokenize import tokenize_for_query
from raysearch.utils import clean_whitespace

if TYPE_CHECKING:
    from raysearch.components.rank.base import RankerBase
    from raysearch.models.steps.research import (
        ResearchStepContext,
        RoundStepContext,
    )


def resolve_research_model(
    *,
    settings: AppSettings,
    stage: str,
    fallback: str,
) -> str:
    """Resolve the appropriate research model for a given stage."""
    model_settings = settings.research.models
    stage_to_model = {
        "plan": model_settings.plan,
        "link_select": model_settings.link_select,
        "overview": model_settings.abstract_analyze,
        "abstract": model_settings.abstract_analyze,
        "content": model_settings.content_analyze,
        "synthesize": model_settings.synthesize,
        "markdown": model_settings.markdown,
    }
    configured_model = clean_whitespace(stage_to_model.get(stage, ""))
    return configured_model or fallback


def pick_sources_by_ids(
    *,
    sources: list[ResearchSource],
    source_ids: list[int],
) -> list[ResearchSource]:
    """Pick sources by their IDs."""
    source_by_id = {source.source_id: source for source in sources}
    out: list[ResearchSource] = []
    for source_id in source_ids:
        source = source_by_id.get(source_id)
        if source is None:
            continue
        out.append(source)
    return out


def _build_research_source_rank_text(source: ResearchSource) -> str:
    """Build text for ranking a research source."""
    parts = [
        f"title={source.title}",
        f"url={source.url}",
        f"overview={clean_whitespace(source.overview)}",
        f"content={clean_whitespace(source.content)}",
    ]
    return "\n".join(part for part in parts if part.strip())


async def rerank_research_sources(
    *,
    ctx: ResearchStepContext | RoundStepContext,
    ranker: RankerBase,
    sources: list[ResearchSource],
    query: str,
) -> list[ResearchSource]:
    """Rerank research sources by relevance."""
    if len(sources) <= 1:
        return list(sources)
    effective_query = clean_whitespace(query) or clean_whitespace(ctx.task.question)
    if not effective_query:
        return list(sources)
    texts = [_build_research_source_rank_text(source) for source in sources]
    try:
        scores = await ranker.score_texts(
            texts,
            query=effective_query,
            query_tokens=tokenize_for_query(effective_query),
            mode="rerank",
        )
    except Exception:
        return list(sources)
    ordered = sorted(
        enumerate(sources),
        key=lambda item: (
            -_score_at(scores=scores, idx=item[0]),
            -float(ctx.knowledge.source_scores.get(item[1].source_id, 0.0)),
            -int(item[1].round_index),
            int(item[1].source_id),
            item[0],
        ),
    )
    return [source for _, source in ordered]


def _score_at(*, scores: list[float], idx: int) -> float:
    return float(scores[idx]) if idx < len(scores) else 0.0


__all__ = [
    "pick_sources_by_ids",
    "rerank_research_sources",
    "resolve_research_model",
]
