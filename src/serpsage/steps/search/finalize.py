from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import SearchRankedCandidate, SearchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class SearchFinalizeStep(StepBase[SearchStepContext]):
    span_name = "step.search_finalize"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        """Finalize output list from precomputed rank candidates.

        Args:
            ctx: Search context that should already contain `ctx.rank.candidates`.
            span: Telemetry span for final ordering and truncation metrics.

        Returns:
            Search context with `ctx.output.results` finalized.
        """
        if bool(ctx.deep.aborted):
            ctx.output.results = []
            span.set_attr("aborted", True)
            span.set_attr("before_count", 0)
            span.set_attr("after_count", 0)
            return ctx

        ranked = list(ctx.rank.candidates or [])
        if not ranked:
            ranked = self._build_fallback_candidates(ctx)

        before_count = len(ranked) + int(ctx.rank.filtered_count or 0)
        ranked = self._sort_candidates(
            ranked,
            enable_sort=bool(ctx.rank.has_sort_feature or ctx.rank.deep_enabled),
        )
        max_results = self._resolve_max_results(ctx)
        ctx.output.results = [item.result for item in ranked[:max_results]]

        span.set_attr("aborted", False)
        span.set_attr("before_count", int(before_count))
        span.set_attr("filtered_by_text", int(ctx.rank.filtered_count or 0))
        span.set_attr("after_count", int(len(ctx.output.results)))
        span.set_attr("max_results", int(max_results))
        span.set_attr("deep_enabled", bool(ctx.rank.deep_enabled))
        return ctx

    def _resolve_max_results(self, ctx: SearchStepContext) -> int:
        if int(ctx.rank.max_results or 0) > 0:
            return int(ctx.rank.max_results)
        return max(1, int(ctx.request.max_results or self.settings.search.max_results))

    def _sort_candidates(
        self,
        candidates: list[SearchRankedCandidate],
        *,
        enable_sort: bool,
    ) -> list[SearchRankedCandidate]:
        out = list(candidates)
        if enable_sort:
            out.sort(key=lambda item: (-float(item.final_score), int(item.order)))
            return out
        out.sort(key=lambda item: int(item.order))
        return out

    def _build_fallback_candidates(
        self, ctx: SearchStepContext
    ) -> list[SearchRankedCandidate]:
        if ctx.output.results:
            return [
                SearchRankedCandidate(result=item, order=idx)
                for idx, item in enumerate(ctx.output.results)
            ]
        return [
            SearchRankedCandidate(result=item.result, order=idx)
            for idx, item in enumerate(ctx.fetch.candidates)
        ]


__all__ = ["SearchFinalizeStep"]
