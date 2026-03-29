from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from serpsage.dependencies import Depends
from serpsage.models.steps.answer import (
    AnswerStepContext,
    AnswerSubQuestionPlan,
)
from serpsage.models.steps.answer.payloads import (
    AnswerPlanPayload,
    AnswerSubQuestionPayload,
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
                request_id=ctx.request_id,
                now_utc=now_utc,
                select_engines=bool(routes),
                engine_selection_context=engine_selection_context,
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="answer.plan.failed",
                request_id=ctx.request_id,
                step="answer.plan",
                error_code="answer_plan_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
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
        request_id: str,
        now_utc: datetime,
        select_engines: bool,
        engine_selection_context: str,
    ) -> AnswerPlanPayload:
        search_query_schema = self._build_search_query_schema(
            select_engines=select_engines
        )
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "answer_mode",
                "freshness_intent",
                "query_language",
                "planner_action",
                "tool_calls",
            ],
            "properties": {
                "answer_mode": {"type": "string", "enum": ["direct", "summary"]},
                "freshness_intent": {"type": "boolean"},
                "query_language": {"type": "string", "minLength": 1},
                "planner_action": {
                    "type": "string",
                    "enum": ["search", "respond"],
                },
                "direct_answer": {"type": "string"},
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["tool_name", "arguments"],
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "enum": ["search"],
                            },
                            "arguments": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["question", "search_query"],
                                "properties": {
                                    "question": {"type": "string", "minLength": 1},
                                    "search_query": search_query_schema,
                                },
                            },
                        },
                    },
                    "minItems": 0,
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
            response_format=schema,
        )
        await self.meter.record(
            name="llm.tokens",
            request_id=request_id,
            model=str(self.settings.answer.plan.use_model),
            unit="token",
            tokens={
                "prompt_tokens": int(result.usage.prompt_tokens),
                "completion_tokens": int(result.usage.completion_tokens),
                "total_tokens": int(result.usage.total_tokens),
            },
        )
        raw = result.data
        if not isinstance(raw, dict):
            raise TypeError("answer planner output must be a JSON object")
        await self.tracker.debug(
            name="answer.plan.detail",
            request_id=request_id,
            step="answer.plan",
            data={
                "planner_action": clean_whitespace(
                    str(raw.get("planner_action") or "")
                ),
                "tool_call_count": len(raw.get("tool_calls") or []),
                "has_direct_answer": bool(
                    clean_whitespace(str(raw.get("direct_answer") or ""))
                ),
            },
        )
        return self._normalize_plan(raw=raw, original_query=query)

    def _build_search_query_schema(self, *, select_engines: bool) -> dict[str, Any]:
        if not select_engines:
            return {"type": "string", "minLength": 1}
        return {
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
        }

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
                        "Planner agent protocol:",
                        "- You operate like a tiny agent with one available tool named search.",
                        "- search tool output must be returned via tool_calls only; do not narrate intermediate reasoning.",
                        "- If search is required, set planner_action=search and return 1-8 search tool calls.",
                        "- If you can answer from stable world knowledge without searching, set planner_action=respond, set tool_calls=[], and provide direct_answer.",
                        "- direct_answer is allowed only for extremely stable, widely known, non-current facts that you can answer with very high confidence.",
                        "- if the question may depend on recency, niche knowledge, ambiguity resolution, evidence gathering, or nontrivial synthesis, do not use direct_answer; use search tool calls instead.",
                        "Decision policy:",
                        "- answer_mode is defined by the expected final response style, not by superficial question length.",
                        "- choose direct only when the final response should be a minimal factual answer (typically short entity/number/date/yes-no).",
                        "- choose summary when the final response needs explanation, comparison, recommendation, tradeoffs, or synthesis.",
                        "- freshness_intent: true ONLY when recency/currentness is explicitly requested.",
                        "- query_language: language+script for final answer output.",
                        "- Use the smallest number of search tool calls that still gives strong retrieval coverage.",
                        "- Exactly one search tool call is allowed only when the question is both atomic factual and genuinely simple, or another genuinely simple question with a narrow answer space that one precise search can settle.",
                        "- Do not use exactly one broad search tool call for open-ended, comparative, explanatory, recommendation, tradeoff, or synthesis questions.",
                        "- For open-ended or multi-facet questions, prefer 2-4 search tool calls from complementary retrieval angles.",
                        "- Use complementary angles such as definition/current state, key drivers or mechanism, evidence/examples, comparisons/options, risks/tradeoffs, or edge cases when relevant.",
                        "- Do not collapse a complex question into one broad search tool call when multiple focused searches would retrieve better evidence.",
                        "- Avoid filler or near-duplicate paraphrases; each search tool call must contribute distinct retrieval value.",
                        "- Keep search tool calls in logical reasoning order.",
                        (
                            "ENGINE_SELECTION_OUTPUT_RULES:\n"
                            "- Each search tool call arguments object must contain question and search_query.\n"
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
                        "- planner_action: search|respond",
                        "- direct_answer: string; required only when planner_action=respond",
                        (
                            "- tool_calls: [{tool_name:'search', arguments:{question, search_query}}] (0-8)"
                            if select_engines
                            else "- tool_calls: [{tool_name:'search', arguments:{question, search_query}}] (0-8), where search_query is a string"
                        ),
                    ]
                ),
            },
        ]

    def _normalize_plan(
        self,
        *,
        raw: object,
        original_query: str,
    ) -> AnswerPlanPayload:
        if isinstance(raw, AnswerPlanPayload):
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
        if not isinstance(raw, dict):
            raise TypeError("answer planner output must be an object")
        query_language = clean_whitespace(str(raw.get("query_language") or ""))
        if not query_language:
            query_language = "same as query"
        answer_mode = clean_whitespace(str(raw.get("answer_mode") or "")).casefold()
        if answer_mode not in {"direct", "summary"}:
            answer_mode = "summary"
        normalized_answer_mode: Literal["direct", "summary"] = (
            "direct" if answer_mode == "direct" else "summary"
        )
        sub_questions = self._extract_sub_questions(raw)
        sub_questions = _normalize_sub_questions(
            sub_questions,
            fallback=clean_whitespace(original_query),
            limit=self._MAX_SUB_QUESTIONS,
        )
        return AnswerPlanPayload(
            answer_mode=normalized_answer_mode,
            freshness_intent=_coerce_bool(raw.get("freshness_intent"), default=False),
            query_language=query_language,
            sub_questions=sub_questions,
        )

    def _extract_sub_questions(
        self,
        raw: dict[str, Any],
    ) -> list[AnswerSubQuestionPayload]:
        tool_calls = raw.get("tool_calls")
        out = self._normalize_tool_calls(tool_calls)
        if out:
            return out
        legacy_sub_questions = raw.get("sub_questions")
        if not isinstance(legacy_sub_questions, list):
            return []
        legacy_out: list[AnswerSubQuestionPayload] = []
        for item in legacy_sub_questions:
            payload = self._try_validate_sub_question(item)
            if payload is None:
                continue
            legacy_out.append(payload)
        return legacy_out

    def _normalize_tool_calls(
        self,
        raw_tool_calls: object,
    ) -> list[AnswerSubQuestionPayload]:
        if not isinstance(raw_tool_calls, list):
            return []
        out: list[AnswerSubQuestionPayload] = []
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            tool_name = clean_whitespace(
                str(item.get("tool_name") or item.get("name") or item.get("tool") or "")
            ).casefold()
            if tool_name and tool_name != "search":
                continue
            raw_arguments = item.get("arguments")
            if raw_arguments is None:
                raw_arguments = item.get("args")
            if not isinstance(raw_arguments, dict):
                continue
            raw_search_query = raw_arguments.get("search_query")
            if raw_search_query is None:
                raw_search_query = raw_arguments.get("query")
            question = clean_whitespace(str(raw_arguments.get("question") or ""))
            if not question:
                if isinstance(raw_search_query, str):
                    question = clean_whitespace(raw_search_query)
                elif isinstance(raw_search_query, dict):
                    question = clean_whitespace(
                        str(raw_search_query.get("query") or "")
                    )
            search_query_value = raw_search_query or question
            if not question or not search_query_value:
                continue
            search_query = self._try_validate_search_query(search_query_value)
            if search_query is None:
                continue
            out.append(
                AnswerSubQuestionPayload(
                    question=question,
                    search_query=search_query,
                )
            )
        return out

    def _try_validate_sub_question(
        self,
        value: object,
    ) -> AnswerSubQuestionPayload | None:
        try:
            return AnswerSubQuestionPayload.model_validate(value)
        except Exception:  # noqa: BLE001
            return None

    def _try_validate_search_query(
        self,
        value: object,
    ) -> QuerySourceSpec | None:
        try:
            return QuerySourceSpec.model_validate(value)
        except Exception:  # noqa: BLE001
            return None


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


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = clean_whitespace(value).casefold()
        if token in {"1", "true", "yes", "y"}:
            return True
        if token in {"0", "false", "no", "n", ""}:
            return False
    return default


__all__ = ["AnswerPlanStep"]
