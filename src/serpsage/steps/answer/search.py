from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchRequestBase,
    FetchSubpagesRequest,
    SearchRequest,
)
from serpsage.models.pipeline import AnswerStepContext, SearchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.steps.base import RunnerBase


_DEEP_SUBPAGE_MAX = 2


class AnswerSearchStep(StepBase[AnswerStepContext]):
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
    async def run_inner(self, ctx: AnswerStepContext) -> AnswerStepContext:

        search_request = self._build_search_request(ctx)

        ctx.search.request = search_request

        ctx.search.search_mode = search_request.mode

        search_ctx = SearchStepContext(
            settings=ctx.settings,
            request=search_request,
            disable_internal_llm=True,
            request_id=ctx.request_id,
        )

        try:
            search_ctx = await self._search_runner.run(search_ctx)

        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="answer.search.error",
                request_id=ctx.request_id,
                stage="search",
                status="error",
                error_code="answer_search_failed",
                error_type=type(exc).__name__,
                attrs={
                    "request_id": ctx.request_id,
                    "message": str(exc),
                },
            )

            ctx.search.results = []

            return ctx

        ctx.search.search_mode = str(search_ctx.request.mode)

        ctx.search.results = list(search_ctx.output.results or [])

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
