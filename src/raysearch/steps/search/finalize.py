from __future__ import annotations

from typing_extensions import override

from raysearch.models.steps.search import SearchRankedCandidate, SearchStepContext
from raysearch.steps.base import StepBase


class SearchFinalizeStep(StepBase[SearchStepContext]):
    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        """Finalize output list from precomputed rank candidates.
        Args:
            ctx: Search context that should already contain `ctx.rank.candidates`.
        Returns:
            Search context with `ctx.output.results` finalized.
        """
        if bool(ctx.plan.aborted):
            ctx.output.results = []
            return ctx
        ranked = list(ctx.rank.candidates or [])
        if not ranked:
            ranked = self._build_fallback_candidates(ctx)
        ranked = self._sort_candidates(
            ranked,
            enable_sort=bool(ctx.rank.has_sort_feature or ctx.rank.use_context_score),
        )
        max_results = self._resolve_max_results(ctx)
        ctx.output.results = [item.result for item in ranked[:max_results]]
        await self.tracker.info(
            name="search.finalize.completed",
            request_id=ctx.request_id,
            step="search.finalize",
            data={
                "result_count": len(ctx.output.results),
            },
        )
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
