from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.steps.research.search import source_authority_score
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.rank.base import RankerBase
    from serpsage.models.steps.research import ResearchSource, ResearchStepContext

_MAX_CONTENT_CHARS = 4000
_MAX_OVERVIEW_CHARS = 1800


def build_research_source_rank_text(source: ResearchSource) -> str:
    parts = [
        f"title={clean_whitespace(source.title)}",
        f"url={clean_whitespace(source.url)}",
        f"overview={clean_whitespace(source.overview)[:_MAX_OVERVIEW_CHARS]}",
        f"content={clean_whitespace(source.content)[:_MAX_CONTENT_CHARS]}",
    ]
    return "\n".join(part for part in parts if part.strip())


async def rerank_research_sources(
    *,
    ctx: ResearchStepContext,
    ranker: RankerBase,
    sources: list[ResearchSource],
    query: str,
) -> list[ResearchSource]:
    if len(sources) <= 1:
        return list(sources)
    effective_query = clean_whitespace(query) or clean_whitespace(ctx.task.question)
    if not effective_query:
        return list(sources)
    texts = [build_research_source_rank_text(source) for source in sources]
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
            -float(source_authority_score(item[1])),
            int(item[1].source_id),
            item[0],
        ),
    )
    return [source for _, source in ordered]


def _score_at(*, scores: list[float], idx: int) -> float:
    return float(scores[idx]) if idx < len(scores) else 0.0


__all__ = ["build_research_source_rank_text", "rerank_research_sources"]
