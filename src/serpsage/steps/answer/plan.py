from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from serpsage.dependencies import Depends
from serpsage.models.steps.answer import (
    AnswerPlanPayload,
    AnswerStepContext,
    AnswerSubQuestionPayload,
    AnswerSubQuestionPlan,
)
from serpsage.models.steps.search import QuerySourceSpec
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace


class AnswerPlanStep(StepBase[AnswerStepContext]):
    _MAX_SUB_QUESTIONS = 8
    _FIXED_MAX_RESULTS = 5
    llm: LLMClientBase = Depends()
    provider: SearchProviderBase = Depends()

    @override
    async def run_inner(self, ctx: AnswerStepContext) -> AnswerStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        query_text = clean_whitespace(ctx.request.query)
        routes = resolve_engine_selection_routes(
            settings=self.settings,
            subsystem="answer",
            provider=self.provider,
        )
        engine_selection_context = build_engine_selection_context(routes=routes)
        try:
            plan = await self._plan_search(
                query=query_text,
                now_utc=now_utc,
                select_engines=bool(routes),
                engine_selection_context=engine_selection_context,
            )
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="answer.plan.error",
                request_id=ctx.request_id,
                stage="plan",
                status="error",
                error_code="answer_plan_failed",
                error_type=type(exc).__name__,
                attrs={
                    "request_id": ctx.request_id,
                    "message": str(exc),
                },
            )
            plan = AnswerPlanPayload(
                answer_mode="summary",
                freshness_intent=False,
                query_language="same as query",
                sub_questions=[
                    AnswerSubQuestionPayload(
                        question=query_text,
                        search_query=QuerySourceSpec(query=query_text),
                    )
                ],
            )
        sub_questions = _normalize_sub_questions(
            plan.sub_questions,
            fallback=query_text,
            limit=self._MAX_SUB_QUESTIONS,
        )
        sub_question_plans = [
            AnswerSubQuestionPlan(
                question=item.question,
                search_query=item.search_query.model_copy(deep=True),
            )
            for item in sub_questions
        ]
        first_search_query = (
            sub_question_plans[0].search_query
            if sub_question_plans
            else QuerySourceSpec(query=query_text)
        )
        ctx.plan.answer_mode = str(plan.answer_mode)
        ctx.plan.freshness_intent = bool(plan.freshness_intent)
        ctx.plan.query_language = str(plan.query_language)
        ctx.plan.search_query = first_search_query
        ctx.plan.search_mode = "auto"
        ctx.plan.max_results = self._FIXED_MAX_RESULTS
        ctx.plan.additional_queries = None
        ctx.plan.sub_questions = [
            item.model_copy(deep=True) for item in sub_question_plans
        ]
        return ctx

    async def _plan_search(
        self,
        *,
        query: str,
        now_utc: datetime,
        select_engines: bool,
        engine_selection_context: str,
    ) -> AnswerPlanPayload:
        sub_question_schema: dict[str, Any]
        if select_engines:
            sub_question_schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "search_query"],
                "properties": {
                    "question": {"type": "string", "minLength": 1},
                    "search_query": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query", "include_sources"],
                        "properties": {
                            "query": {"type": "string", "minLength": 1},
                            "include_sources": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            }
        else:
            sub_question_schema = {"type": "string", "minLength": 1}
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "answer_mode",
                "freshness_intent",
                "query_language",
                "sub_questions",
            ],
            "properties": {
                "answer_mode": {"type": "string", "enum": ["direct", "summary"]},
                "freshness_intent": {"type": "boolean"},
                "query_language": {"type": "string", "minLength": 1},
                "sub_questions": {
                    "type": "array",
                    "items": sub_question_schema,
                    "minItems": 1,
                    "maxItems": self._MAX_SUB_QUESTIONS,
                },
            },
        }
        messages = self._build_messages(
            query=query,
            now_utc=now_utc,
            select_engines=select_engines,
            engine_selection_context=engine_selection_context,
        )
        result = await self.llm.create(
            model=str(self.settings.answer.plan.use_model),
            messages=messages,
            response_format=AnswerPlanPayload,
            format_override=schema,
        )
        raw = result.data
        return self._normalize_plan(raw=raw, original_query=query)

    def _build_messages(
        self,
        *,
        query: str,
        now_utc: datetime,
        select_engines: bool,
        engine_selection_context: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a planner for a web-answer pipeline. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        f"Current UTC timestamp: {now_utc.isoformat()}",
                        f"Current UTC date: {now_utc.date().isoformat()}",
                        f"Question: {query}",
                        "Decision policy:",
                        "- answer_mode is defined by the expected final response style, not by superficial question length.",
                        "- choose direct only when the final response should be a minimal factual answer (typically short entity/number/date/yes-no).",
                        "- choose summary when the final response needs explanation, comparison, recommendation, tradeoffs, or synthesis.",
                        "- freshness_intent: true ONLY when recency/currentness is explicitly requested.",
                        "- query_language: language+script for final answer output.",
                        "- sub_questions: output 1-8 only when decomposition is necessary for retrieval coverage.",
                        "- default to minimal decomposition; do not create filler sub_questions.",
                        "- if one query is enough, output exactly one sub_question.",
                        "- keep sub_questions in logical reasoning order.",
                        (
                            "ENGINE_SELECTION_OUTPUT_RULES:\n"
                            "- Each sub_question should be an object with question and search_query.\n"
                            "- search_query must include query and include_sources.\n"
                            "- question is the readable decomposition; search_query is the retrieval wording and may differ.\n"
                            "- Prefer include_sources=[] unless a narrower route clearly improves retrieval.\n\n"
                            f"{engine_selection_context}"
                        )
                        if engine_selection_context
                        else "",
                        "Output fields:",
                        "- answer_mode: direct|summary",
                        "- freshness_intent: boolean",
                        "- query_language: language+script",
                        (
                            "- sub_questions: {question, search_query}[] (1-8)"
                            if select_engines
                            else "- sub_questions: string[] (1-8)"
                        ),
                    ]
                ),
            },
        ]

    def _normalize_plan(
        self,
        *,
        raw: AnswerPlanPayload,
        original_query: str,
    ) -> AnswerPlanPayload:
        query_language = clean_whitespace(raw.query_language)
        if not query_language:
            query_language = "same as query"
        sub_questions = _normalize_sub_questions(
            list(raw.sub_questions),
            fallback=clean_whitespace(original_query),
            limit=self._MAX_SUB_QUESTIONS,
        )
        return raw.model_copy(
            update={
                "query_language": query_language,
                "sub_questions": sub_questions,
            },
            deep=True,
        )


def _normalize_sub_questions(
    raw_items: list[AnswerSubQuestionPayload],
    *,
    fallback: str,
    limit: int,
) -> list[AnswerSubQuestionPayload]:
    out: list[AnswerSubQuestionPayload] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for item in raw_items:
        question = clean_whitespace(item.question)
        if not question:
            continue
        search_query = item.search_query.model_copy(deep=True)
        key = (
            question.casefold(),
            search_query.query.casefold(),
            tuple(search_query.include_sources),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            AnswerSubQuestionPayload(question=question, search_query=search_query)
        )
        if len(out) >= limit:
            break
    if out:
        return out
    fallback_question = clean_whitespace(fallback)
    return (
        [
            AnswerSubQuestionPayload(
                question=fallback_question,
                search_query=QuerySourceSpec(query=fallback_question),
            )
        ]
        if fallback_question
        else []
    )


__all__ = ["AnswerPlanStep"]
