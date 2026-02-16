from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import SearchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
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
            if r.page and r.page.abstracts:
                for j, abstract in enumerate(r.page.abstracts, 1):
                    abstract.abstract_id = abstract.abstract_id or f"{sid}:A{j}"


__all__ = ["SearchFinalizeStep"]
