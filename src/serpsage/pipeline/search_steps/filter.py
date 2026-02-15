from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import SearchStepContext
from serpsage.pipeline.step import PipelineStep

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime


class FilterStep(PipelineStep[SearchStepContext]):
    span_name = "step.filter"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        span.set_attr("before_count", int(len(ctx.results or [])))
        ctx.results = self._filter(
            results=ctx.results,
            query_tokens=ctx.query_tokens or [],
        )
        span.set_attr("after_count", int(len(ctx.results or [])))
        return ctx

    def _filter(
        self,
        *,
        results: list[ResultItem],
        query_tokens: list[str],
    ) -> list[ResultItem]:
        return [
            r
            for r in results
            if self._is_not_noise(r)
            and self._is_relevant(r, query_tokens=query_tokens)
        ]

    def _is_not_noise(self, r: ResultItem) -> bool:
        title = (r.title or "").strip()
        snippet = (r.snippet or "").strip()

        if not title and not snippet:
            return False

        return not (len(title) < 2 and len(snippet) < 40)

    def _is_relevant(self, r: ResultItem, *, query_tokens: list[str]) -> bool:
        t = (r.title or "").lower()
        s = (r.snippet or "").lower()
        return any(tok in t or tok in s for tok in query_tokens)


__all__ = ["FilterStep"]
