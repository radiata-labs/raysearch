from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchStepContext, ScoredAbstract
from serpsage.pipeline.step import PipelineStep
from serpsage.text.abstracts import (
    AbstractCandidate,
    apply_title_logit_boost,
    extract_abstract_candidates,
    fit_abstract_budget,
)

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import RankerBase
    from serpsage.core.runtime import Runtime


class FetchAbstractRankStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_abstract_rank"

    def __init__(self, *, rt: Runtime, ranker: RankerBase) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker
        self.bind_deps(ranker)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        abstracts_request = ctx.abstracts_request
        span.set_attr("has_abstracts", bool(abstracts_request is not None))
        if abstracts_request is None:
            return ctx

        if ctx.extracted is None:
            ctx.errors.append(
                AppError(
                    code="fetch_abstract_rank_failed",
                    message="missing extracted content",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "rank",
                        "fatal": False,
                        "crawl_mode": ctx.others_runtime.crawl_mode,
                    },
                )
            )
            return ctx

        markdown = str(ctx.extracted.markdown or "")
        if not markdown.strip():
            ctx.errors.append(
                AppError(
                    code="fetch_abstract_rank_failed",
                    message="no content extracted",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "rank",
                        "fatal": False,
                        "crawl_mode": ctx.others_runtime.crawl_mode,
                    },
                )
            )
            return ctx

        abstract_cfg = self.settings.fetch.abstract
        candidates = extract_abstract_candidates(
            markdown=markdown,
            max_markdown_chars=int(abstract_cfg.max_markdown_chars),
            max_abstracts=int(abstract_cfg.max_abstracts),
            min_abstract_chars=int(abstract_cfg.min_abstract_chars),
        )
        if not candidates:
            ctx.errors.append(
                AppError(
                    code="fetch_abstract_rank_failed",
                    message="no abstracts",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "rank",
                        "fatal": False,
                        "crawl_mode": ctx.others_runtime.crawl_mode,
                    },
                )
            )
            return ctx

        query = abstracts_request.query
        base_scores = await self._ranker.score_texts(
            texts=[candidate.text for candidate in candidates],
            query=query,
            query_tokens=list(ctx.abstract_query_tokens or []),
            intent_tokens=list(ctx.abstract_intent_tokens or []),
        )
        heading_scores = await self._score_headings(
            query=query,
            candidates=candidates,
            query_tokens=list(ctx.abstract_query_tokens or []),
            intent_tokens=list(ctx.abstract_intent_tokens or []),
        )
        alpha = float(abstract_cfg.title_boost_alpha)
        min_score = float(abstract_cfg.min_abstract_score)

        scored: list[tuple[float, AbstractCandidate]] = []
        for idx, candidate in enumerate(candidates):
            base = float(base_scores[idx]) if idx < len(base_scores) else 0.0
            title_score = float(heading_scores.get(candidate.heading, 0.0))
            final_score = apply_title_logit_boost(
                abstract_score=base,
                title_score=title_score,
                alpha=alpha,
            )
            if final_score < min_score:
                continue
            scored.append((float(final_score), candidate))

        if not scored:
            ctx.errors.append(
                AppError(
                    code="fetch_abstract_rank_failed",
                    message="no matching abstracts",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "rank",
                        "fatal": False,
                        "crawl_mode": ctx.others_runtime.crawl_mode,
                    },
                )
            )
            return ctx

        scored.sort(key=lambda item: (-item[0], int(item[1].position)))
        top_k = int(
            abstracts_request.top_k_abstracts
            if abstracts_request.top_k_abstracts is not None
            else abstract_cfg.default_top_k_abstracts
        )
        kept = fit_abstract_budget(
            ranked=scored,
            top_k_abstracts=top_k,
            max_chars=abstracts_request.max_chars,
        )
        ctx.scored_abstracts = [
            ScoredAbstract(
                abstract_id=f"S1:A{i + 1}",
                text=item.text,
                score=float(score),
            )
            for i, (score, item) in enumerate(kept)
        ]

        span.set_attr("top_k_abstracts", int(top_k))
        span.set_attr("abstracts_kept", int(len(ctx.scored_abstracts)))
        return ctx

    async def _score_headings(
        self,
        *,
        query: str,
        candidates: list[AbstractCandidate],
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> dict[str, float]:
        headings: list[str] = []
        for candidate in candidates:
            heading = (candidate.heading or "").strip()
            if heading and heading not in headings:
                headings.append(heading)
        if not headings:
            return {}
        scores = await self._ranker.score_texts(
            texts=headings,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        return {
            heading: (float(scores[idx]) if idx < len(scores) else 0.0)
            for idx, heading in enumerate(headings)
        }


__all__ = ["FetchAbstractRankStep"]
