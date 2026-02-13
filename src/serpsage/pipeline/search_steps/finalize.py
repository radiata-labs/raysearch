from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import SearchStepContext
from serpsage.pipeline.step import PipelineStep

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime


class SearchFinalizeStep(PipelineStep[SearchStepContext]):
    span_name = "step.search_finalize"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        min_score = float(self.settings.search.min_score)
        max_results = int(ctx.request.max_results or self.settings.search.max_results)

        before = len(ctx.results)
        ctx.results = [
            r
            for r in ctx.results
            if float(r.score) > 0.0 and float(r.score) >= min_score
        ]
        ctx.results = ctx.results[: max(1, max_results)]
        self._assign_ids(ctx.results)
        span.set_attr("before_count", int(before))
        span.set_attr("after_count", int(len(ctx.results)))
        span.set_attr("min_score", float(min_score))
        span.set_attr("max_results", int(max_results))
        return ctx

    def _assign_ids(self, results: list[ResultItem]) -> None:
        for i, r in enumerate(results, 1):
            sid = f"S{i}"
            r.source_id = sid
            if r.page and r.page.chunks:
                for j, ch in enumerate(r.page.chunks, 1):
                    ch.chunk_id = ch.chunk_id or f"{sid}:C{j}"


__all__ = ["SearchFinalizeStep"]
