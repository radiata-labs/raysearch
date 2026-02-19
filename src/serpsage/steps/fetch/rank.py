from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchStepContext, PreparedAbstract, ScoredAbstract
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace, tokenize, tokenize_for_query

if TYPE_CHECKING:
    from serpsage.components.rank.base import RankerBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class FetchAbstractRankStep(StepBase[FetchStepContext]):
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

        abstracts_req = ctx.abstracts_request
        overview_req = ctx.overview_request
        span.set_attr("has_abstracts", bool(abstracts_req is not None))
        span.set_attr("has_overview", bool(overview_req is not None))
        if abstracts_req is None and overview_req is None:
            return ctx

        candidates = list(ctx.prepared_abstracts or [])
        if not candidates:
            ctx.errors.append(
                AppError(
                    code="fetch_abstract_rank_failed",
                    message="no prepared abstracts",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "rank",
                        "fatal": False,
                        "crawl_mode": ctx.others.crawl_mode,
                    },
                )
            )
            return ctx

        abstracts_query = ""
        raw_scored_abstracts = []
        if abstracts_req is not None:
            abstracts_query = _resolve_effective_query(
                requested_query=abstracts_req.query,
                title=(ctx.extracted.title if ctx.extracted else ""),
                url=ctx.url,
            )
            abstract_budget = (
                int(abstracts_req.max_chars)
                if abstracts_req.max_chars is not None
                else int(self.settings.fetch.abstract.max_abstract_chars)
            )
            raw_scored_abstracts = await self._score_abstracts(
                query=abstracts_query,
                candidates=candidates,
                query_tokens=tokenize_for_query(abstracts_query),
            )

            if not raw_scored_abstracts:
                ctx.errors.append(
                    AppError(
                        code="fetch_abstract_rank_failed",
                        message="no matching abstracts",
                        details={
                            "url": ctx.url,
                            "url_index": ctx.url_index,
                            "stage": "rank",
                            "fatal": False,
                            "crawl_mode": ctx.others.crawl_mode,
                        },
                    )
                )
            else:
                ctx.scored_abstracts = [
                    ScoredAbstract(
                        abstract_id=f"S1:A{i + 1}",
                        text=item.text,
                        score=float(score),
                    )
                    for i, (score, item) in enumerate(
                        _fit_budget(
                            ranked=raw_scored_abstracts, max_chars=abstract_budget
                        )
                    )
                ]

        if overview_req is not None:
            overview_query = _resolve_effective_query(
                requested_query=overview_req.query,
                title=(ctx.extracted.title if ctx.extracted else ""),
                url=ctx.url,
            )
            if (
                abstracts_req is not None
                and bool(ctx.scored_abstracts)
                and overview_query == abstracts_query
                and raw_scored_abstracts
            ):
                ctx.overview_scored_abstracts = [
                    ScoredAbstract(
                        abstract_id=f"S1:A{i + 1}",
                        text=item.text,
                        score=float(score),
                    )
                    for i, (score, item) in enumerate(
                        _fit_budget(
                            ranked=raw_scored_abstracts,
                            max_chars=int(
                                self.settings.fetch.overview.max_abstract_chars
                            ),
                        )
                    )
                ]
            else:
                raw_overview_scored_abstracts = await self._score_abstracts(
                    query=overview_query,
                    candidates=candidates,
                    query_tokens=tokenize_for_query(overview_query),
                )
                ctx.overview_scored_abstracts = [
                    ScoredAbstract(
                        abstract_id=f"S1:A{i + 1}",
                        text=item.text,
                        score=float(score),
                    )
                    for i, (score, item) in enumerate(
                        _fit_budget(
                            ranked=raw_overview_scored_abstracts,
                            max_chars=int(
                                self.settings.fetch.overview.max_abstract_chars
                            ),
                        )
                    )
                ]

        span.set_attr("abstracts_kept", int(len(ctx.scored_abstracts)))
        span.set_attr(
            "overview_abstracts_kept", int(len(ctx.overview_scored_abstracts))
        )
        return ctx

    async def _score_abstracts(
        self,
        *,
        query: str,
        candidates: list[PreparedAbstract],
        query_tokens: list[str],
    ) -> list[tuple[float, PreparedAbstract]]:
        min_tokens = int(self.settings.fetch.abstract.min_abstract_tokens)
        query_token_count = len(query_tokens)
        filtered_candidates: list[PreparedAbstract] = []
        for candidate in candidates:
            token_count = len(tokenize(candidate.text))
            if token_count <= min_tokens:
                continue
            if token_count <= (query_token_count * 2):
                continue
            filtered_candidates.append(candidate)
        candidates = filtered_candidates
        if not candidates:
            return []

        base_scores = await self._ranker.score_texts(
            texts=[candidate.text for candidate in candidates],
            query=query,
            query_tokens=list(query_tokens or []),
        )
        heading_scores = await self._score_headings(
            query=query,
            candidates=candidates,
            query_tokens=list(query_tokens or []),
        )

        cfg = self.settings.fetch.abstract
        alpha = float(cfg.title_boost_alpha)
        min_score = float(cfg.min_abstract_score)

        scored: list[tuple[float, PreparedAbstract]] = []
        for idx, candidate in enumerate(candidates):
            base = float(base_scores[idx]) if idx < len(base_scores) else 0.0
            title_score = float(heading_scores.get(candidate.heading, 0.0))
            final_score = _apply_title_logit_boost(
                abstract_score=base,
                title_score=title_score,
                alpha=alpha,
            )
            if final_score < min_score:
                continue
            scored.append((final_score, candidate))

        if not scored:
            return []

        scored.sort(key=lambda item: (-item[0], int(item[1].position)))

        return scored

    async def _score_headings(
        self,
        *,
        query: str,
        candidates: list[PreparedAbstract],
        query_tokens: list[str],
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
        )
        return {
            heading: (float(scores[idx]) if idx < len(scores) else 0.0)
            for idx, heading in enumerate(headings)
        }


def _resolve_effective_query(
    *, requested_query: str | None, title: str, url: str
) -> str:
    query = clean_whitespace(requested_query or "")
    if query:
        return query
    normalized_title = clean_whitespace(title or "")
    if normalized_title:
        return normalized_title
    return clean_whitespace(url or "")


def _apply_title_logit_boost(
    *,
    abstract_score: float,
    title_score: float,
    alpha: float,
) -> float:
    eps = 1e-6
    sa = min(1.0 - eps, max(eps, float(abstract_score)))
    st = min(1.0 - eps, max(eps, float(title_score)))
    la = math.log(sa / (1.0 - sa))
    lt = max(0.0, math.log(st / (1.0 - st)))
    boosted = 1.0 / (1.0 + math.exp(-(la + float(alpha) * lt)))
    return min(1.0, max(sa, boosted))


def _fit_budget(
    *,
    ranked: list[tuple[float, PreparedAbstract]],
    max_chars: int | None,
) -> list[tuple[float, PreparedAbstract]]:
    if max_chars is None or max_chars <= 0:
        return list(ranked)

    out: list[tuple[float, PreparedAbstract]] = []
    total_chars = 0
    for score, candidate in ranked:
        extra_newline = 1 if out else 0
        next_total = total_chars + extra_newline + len(candidate.text)
        if next_total > int(max_chars):
            break
        total_chars = next_total
        out.append((score, candidate))
    return out


__all__ = ["FetchAbstractRankStep"]
