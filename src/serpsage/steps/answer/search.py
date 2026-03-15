from __future__ import annotations

from typing import Literal
from typing_extensions import override

from serpsage.dependencies import SEARCH_RUNNER, Depends
from serpsage.models.app.request import (
    FetchAbstractsRequest,
    SearchFetchRequest,
    SearchRequest,
)
from serpsage.models.app.response import SearchResponse
from serpsage.models.steps.answer import (
    AnswerStepContext,
    AnswerSubQuestionPlan,
    AnswerSubSearchState,
)
from serpsage.models.steps.search import (
    QuerySourceSpec,
    SearchRuntimeState,
    SearchStepContext,
)
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.utils import clean_whitespace

_MAX_SUB_QUESTIONS = 8
_FIXED_SEARCH_MODE: Literal["auto"] = "auto"
_FIXED_MAX_RESULTS = 5
_FIXED_ABSTRACT_MAX_CHARS = 1000


class AnswerSearchStep(StepBase[AnswerStepContext]):
    search_runner: RunnerBase[SearchStepContext] = Depends(SEARCH_RUNNER)

    @override
    async def run_inner(self, ctx: AnswerStepContext) -> AnswerStepContext:
        sub_questions = self._resolve_sub_questions(ctx)
        requests = [
            self._build_search_request(ctx=ctx, query=item.search_query)
            for item in sub_questions
        ]
        if not requests:
            ctx.search.request = None
            ctx.search.search_mode = _FIXED_SEARCH_MODE
            ctx.search.sub_searches = []
            ctx.search.results = []
            return ctx
        ctx.search.request = requests[0].model_copy(deep=True)
        ctx.search.search_mode = _FIXED_SEARCH_MODE
        search_contexts = [
            SearchStepContext(
                request=req,
                response=SearchResponse(
                    request_id=ctx.request_id,
                    search_mode=req.mode,
                    results=[],
                ),
                runtime=SearchRuntimeState(
                    disable_internal_llm=True,
                    engine_selection_subsystem="answer",
                ),
                request_id=ctx.request_id,
            )
            for req in requests
        ]
        try:
            search_contexts = await self.search_runner.run_batch(search_contexts)
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="answer.search.failed",
                request_id=ctx.request_id,
                step="answer.search",
                error_code="answer_search_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "sub_question_count": len(sub_questions),
                },
            )
            ctx.search.sub_searches = [
                AnswerSubSearchState(
                    question=item.question,
                    search_query=item.search_query,
                    request=requests[idx].model_copy(deep=True),
                    search_mode=_FIXED_SEARCH_MODE,
                    results=[],
                )
                for idx, item in enumerate(sub_questions)
            ]
            ctx.search.results = []
            return ctx
        sub_searches: list[AnswerSubSearchState] = []
        merged_results = []
        for idx, item in enumerate(sub_questions):
            req = requests[idx]
            sub_ctx = search_contexts[idx] if idx < len(search_contexts) else None
            results = list(sub_ctx.output.results or []) if sub_ctx is not None else []
            merged_results.extend(results)
            sub_searches.append(
                AnswerSubSearchState(
                    question=item.question,
                    search_query=item.search_query,
                    request=req.model_copy(deep=True),
                    search_mode=_FIXED_SEARCH_MODE,
                    results=results,
                )
            )
        ctx.search.search_mode = _FIXED_SEARCH_MODE
        ctx.search.sub_searches = sub_searches
        ctx.search.results = merged_results
        return ctx

    def _resolve_sub_questions(
        self, ctx: AnswerStepContext
    ) -> list[AnswerSubQuestionPlan]:
        out: list[AnswerSubQuestionPlan] = []
        for item in list(ctx.plan.sub_questions or []):
            question = clean_whitespace(str(item.question or ""))
            search_query = item.search_query or QuerySourceSpec(query=question)
            if not question or not search_query:
                continue
            out.append(
                AnswerSubQuestionPlan(
                    question=question,
                    search_query=search_query.model_copy(deep=True),
                )
            )
            if len(out) >= _MAX_SUB_QUESTIONS:
                break
        if out:
            return out
        fallback = clean_whitespace(
            (ctx.plan.search_query.query if ctx.plan.search_query is not None else "")
            or ctx.request.query
        )
        if not fallback:
            return []
        return [
            AnswerSubQuestionPlan(
                question=fallback,
                search_query=QuerySourceSpec(query=fallback),
            )
        ]

    def _build_search_request(
        self, *, ctx: AnswerStepContext, query: QuerySourceSpec | None
    ) -> SearchRequest:
        query_text = query.query if query is not None else ""
        return SearchRequest(
            query=query_text,
            user_location="US",
            mode=_FIXED_SEARCH_MODE,
            max_results=_FIXED_MAX_RESULTS,
            additional_queries=None,
            fetchs=SearchFetchRequest(
                content=bool(ctx.request.content),
                abstracts=FetchAbstractsRequest(
                    query=query_text,
                    max_chars=_FIXED_ABSTRACT_MAX_CHARS,
                ),
                subpages=None,
                overview=False,
            ),
        )


__all__ = ["AnswerSearchStep"]
