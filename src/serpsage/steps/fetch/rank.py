from __future__ import annotations

import math
from typing_extensions import override

from serpsage.components.fetch.base import FetchConfigBase
from serpsage.components.rank.base import RankerBase
from serpsage.core.runtime import Runtime
from serpsage.dependencies import Inject
from serpsage.models.steps.fetch import (
    FetchStepContext,
    PreparedPassage,
    ScoredPassage,
)
from serpsage.steps.base import StepBase
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import clean_whitespace


class FetchAbstractRankStep(StepBase[FetchStepContext]):
    def __init__(
        self, *, rt: Runtime = Inject(), ranker: RankerBase = Inject()
    ) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker
        self.bind_deps(ranker)

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        abstracts_req = ctx.analysis.abstracts.request
        overview_req = ctx.analysis.overview.request
        if abstracts_req is None and overview_req is None:
            return ctx
        candidates = list(ctx.analysis.abstracts.prepared or [])
        if not candidates:
            await self.emit_tracking_event(
                event_name="fetch.rank.error",
                request_id=ctx.request_id,
                stage="rank",
                status="error",
                error_code="fetch_abstract_rank_failed",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": False,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": "no prepared abstracts",
                },
            )
            return ctx
        abstracts_query = ""
        raw_scored = []
        fetch_cfg = self.components.resolve_default_config(
            "fetch", expected_type=FetchConfigBase
        )
        if abstracts_req is not None:
            abstracts_query = _resolve_effective_query(
                requested_query=abstracts_req.query,
                title=(ctx.page.doc.meta.title if ctx.page.doc else ""),
                url=ctx.url,
            )
            abstract_budget = (
                int(abstracts_req.max_chars)
                if abstracts_req.max_chars is not None
                else int(fetch_cfg.abstract.max_abstract_chars)
            )
            raw_scored = await self._score_passages(
                query=abstracts_query,
                candidates=candidates,
                query_tokens=tokenize_for_query(abstracts_query),
            )
            if not raw_scored:
                await self.emit_tracking_event(
                    event_name="fetch.rank.error",
                    request_id=ctx.request_id,
                    stage="rank",
                    status="error",
                    error_code="fetch_abstract_rank_failed",
                    attrs={
                        "url": ctx.url,
                        "url_index": int(ctx.url_index),
                        "fatal": False,
                        "crawl_mode": str(ctx.page.crawl_mode),
                        "message": "no matching abstracts",
                    },
                )
            else:
                ctx.analysis.abstracts.ranked = [
                    ScoredPassage(
                        passage_id=f"S1:A{i + 1}",
                        text=item.text,
                        score=float(score),
                    )
                    for i, (score, item) in enumerate(
                        _fit_budget(ranked=raw_scored, max_chars=abstract_budget)
                    )
                ]
        if overview_req is not None:
            overview_query = _resolve_effective_query(
                requested_query=overview_req.query,
                title=(ctx.page.doc.meta.title if ctx.page.doc else ""),
                url=ctx.url,
            )
            if (
                abstracts_req is not None
                and bool(ctx.analysis.abstracts.ranked)
                and overview_query == abstracts_query
                and raw_scored
            ):
                ctx.analysis.overview.ranked = [
                    ScoredPassage(
                        passage_id=f"S1:A{i + 1}",
                        text=item.text,
                        score=float(score),
                    )
                    for i, (score, item) in enumerate(
                        _fit_budget(
                            ranked=raw_scored,
                            max_chars=int(fetch_cfg.overview.max_abstract_chars),
                        )
                    )
                ]
            else:
                raw_overview = await self._score_passages(
                    query=overview_query,
                    candidates=candidates,
                    query_tokens=tokenize_for_query(overview_query),
                )
                ctx.analysis.overview.ranked = [
                    ScoredPassage(
                        passage_id=f"S1:A{i + 1}",
                        text=item.text,
                        score=float(score),
                    )
                    for i, (score, item) in enumerate(
                        _fit_budget(
                            ranked=raw_overview,
                            max_chars=int(fetch_cfg.overview.max_abstract_chars),
                        )
                    )
                ]
        return ctx

    async def _score_passages(
        self,
        *,
        query: str,
        candidates: list[PreparedPassage],
        query_tokens: list[str],
    ) -> list[tuple[float, PreparedPassage]]:
        if not candidates:
            return []
        headings: list[str] = []
        for candidate in candidates:
            heading = clean_whitespace(candidate.heading or "")
            if heading and heading not in headings:
                headings.append(heading)
        combined_texts = [candidate.text for candidate in candidates] + headings
        combined_scores = await self._ranker.score_texts(
            combined_texts,
            query=query,
            query_tokens=list(query_tokens or []),
        )
        base_scores = combined_scores[: len(candidates)]
        heading_scores = {
            heading: (
                float(combined_scores[len(candidates) + idx])
                if len(candidates) + idx < len(combined_scores)
                else 0.0
            )
            for idx, heading in enumerate(headings)
        }
        cfg = self.components.resolve_default_config(
            "fetch", expected_type=FetchConfigBase
        ).abstract
        alpha = float(cfg.title_boost_alpha)
        min_score = float(cfg.min_abstract_score)
        scored: list[tuple[float, PreparedPassage]] = []
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
        scored.sort(key=lambda item: (-item[0], int(item[1].order)))
        return scored


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
    ranked: list[tuple[float, PreparedPassage]],
    max_chars: int | None,
) -> list[tuple[float, PreparedPassage]]:
    if max_chars is None or max_chars <= 0:
        return list(ranked)
    out: list[tuple[float, PreparedPassage]] = []
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
