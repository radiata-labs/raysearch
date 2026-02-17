from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchStepContext, PreparedAbstract, ScoredAbstract
from serpsage.steps.base import StepBase

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
        req = ctx.abstracts_request
        span.set_attr("has_abstracts", bool(req is not None))
        if req is None:
            return ctx

        candidates = list(ctx.prepared_abstracts or [])
        print(candidates)
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

        query = req.query
        base_scores = await self._ranker.score_texts(
            texts=[candidate.text for candidate in candidates],
            query=query,
            query_tokens=list(ctx.abstract_query_tokens or []),
        )
        heading_scores = await self._score_headings(
            query=query,
            candidates=candidates,
            query_tokens=list(ctx.abstract_query_tokens or []),
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
            return ctx

        scored.sort(key=lambda item: (-item[0], int(item[1].position)))
        top_k = int(
            req.top_k_abstracts
            if req.top_k_abstracts is not None
            else cfg.default_top_k_abstracts
        )
        kept = _fit_budget(
            ranked=scored,
            top_k_abstracts=top_k,
            max_chars=req.max_chars,
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
    top_k_abstracts: int,
    max_chars: int | None,
) -> list[tuple[float, PreparedAbstract]]:
    out: list[tuple[float, PreparedAbstract]] = []
    total_chars = 0
    limit = max(1, int(top_k_abstracts))
    for score, candidate in ranked:
        if len(out) >= limit:
            break
        if max_chars is not None and max_chars > 0:
            extra_newline = 1 if out else 0
            next_total = total_chars + extra_newline + len(candidate.text)
            if next_total > int(max_chars):
                break
            total_chars = next_total
        out.append((score, candidate))
    return out


__all__ = ["FetchAbstractRankStep"]
