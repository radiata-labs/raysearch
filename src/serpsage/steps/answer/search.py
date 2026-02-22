from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchRequestBase,
    FetchSubpagesRequest,
    SearchRequest,
)
from serpsage.models.errors import AppError
from serpsage.models.pipeline import AnswerStepContext, SearchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.steps.base import RunnerBase
    from serpsage.telemetry.base import SpanBase


_DEEP_SUBPAGE_MAX = 2


class AnswerSearchStep(StepBase[AnswerStepContext]):
    span_name = "step.answer_search"

    def __init__(
        self,
        *,
        rt: Runtime,
        search_runner: RunnerBase[SearchStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._search_runner = search_runner
        self.bind_deps(search_runner)

    @override
    async def run_inner(
        self, ctx: AnswerStepContext, *, span: SpanBase
    ) -> AnswerStepContext:
        search_request = self._build_search_request(ctx)
        ctx.search.request = search_request
        ctx.search.search_mode = search_request.mode

        search_ctx = SearchStepContext(
            settings=ctx.settings,
            request=search_request,
            plan=ctx.plan,  # Pass plan for reuse in search pipeline
            request_id=ctx.request_id,
        )
        try:
            search_ctx = await self._search_runner.run(search_ctx)
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="answer_search_failed",
                    message=str(exc),
                    details={
                        "request_id": ctx.request_id,
                        "stage": "search",
                    },
                )
            )
            ctx.search.results = []
            span.set_attr("results_count", 0)
            span.set_attr("search_error", True)
            return ctx

        ctx.search.search_mode = str(search_ctx.request.mode)
        ctx.search.results = list(search_ctx.output.results or [])
        ctx.errors.extend(search_ctx.errors)
        span.set_attr("results_count", int(len(ctx.search.results)))
        span.set_attr("search_error", False)
        return ctx

    def _build_search_request(self, ctx: AnswerStepContext) -> SearchRequest:
        mode = str(ctx.plan.search_mode or "auto")
        subpages = (
            FetchSubpagesRequest(
                max_subpages=_DEEP_SUBPAGE_MAX,
                subpage_keywords=ctx.plan.search_query,
            )
            if mode == "deep"
            else None
        )
        return SearchRequest(
            query=ctx.plan.search_query,
            mode="deep" if mode == "deep" else "auto",
            max_results=int(ctx.plan.max_results or self.settings.search.max_results),
            additional_queries=ctx.plan.additional_queries,
            fetchs=FetchRequestBase(
                content=bool(ctx.request.content),
                abstracts=FetchAbstractsRequest(query=ctx.plan.search_query),
                subpages=subpages,
                overview=False,
            ),
        )


__all__ = ["AnswerSearchStep"]
