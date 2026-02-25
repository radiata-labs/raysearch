from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import AnswerStepContext
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


_MAX_ADDITIONAL_QUERIES = 4
_ANSWER_MODE = Literal["direct", "summary"]


@dataclass(slots=True)
class _PlannedSearch:
    answer_mode: _ANSWER_MODE
    freshness_intent: bool
    query_language: str
    search_query: str
    search_mode: Literal["auto", "deep"]
    max_results: int
    additional_queries: list[str] | None


class AnswerPlanStep(StepBase[AnswerStepContext]):
    span_name = "step.answer_plan"

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(
        self, ctx: AnswerStepContext, *, span: SpanBase
    ) -> AnswerStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        max_results_cap = max(1, int(self.settings.search.max_results))

        try:
            plan = await self._plan_search(
                query=ctx.request.query,
                now_utc=now_utc,
                max_results_cap=max_results_cap,
            )
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="answer_plan_failed",
                    message=str(exc),
                    details={"request_id": ctx.request_id, "stage": "plan"},
                )
            )
            plan = _PlannedSearch(
                answer_mode="summary",
                freshness_intent=False,
                query_language="same as query",
                search_query=clean_whitespace(ctx.request.query),
                search_mode="auto",
                max_results=min(max_results_cap, 5),
                additional_queries=None,
            )

        ctx.plan.answer_mode = str(plan.answer_mode)
        ctx.plan.freshness_intent = bool(plan.freshness_intent)
        ctx.plan.query_language = str(plan.query_language)
        ctx.plan.search_query = plan.search_query
        ctx.plan.search_mode = plan.search_mode
        ctx.plan.max_results = int(plan.max_results)
        ctx.plan.additional_queries = (
            list(plan.additional_queries)
            if plan.additional_queries is not None
            else None
        )

        span.set_attr("answer_mode", str(ctx.plan.answer_mode))
        span.set_attr("freshness_intent", bool(ctx.plan.freshness_intent))
        span.set_attr("query_language", str(ctx.plan.query_language))
        span.set_attr("search_mode", str(ctx.plan.search_mode))
        span.set_attr("max_results", int(ctx.plan.max_results))
        span.set_attr(
            "additional_queries_count", int(len(ctx.plan.additional_queries or []))
        )
        span.set_attr(
            "has_plan_error",
            bool(any(err.code == "answer_plan_failed" for err in ctx.errors)),
        )
        return ctx

    async def _plan_search(
        self,
        *,
        query: str,
        now_utc: datetime,
        max_results_cap: int,
    ) -> _PlannedSearch:
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "answer_mode",
                "freshness_intent",
                "query_language",
                "search_query",
                "optimize_query",
                "search_mode",
                "max_results",
                "additional_queries",
            ],
            "properties": {
                "answer_mode": {"type": "string", "enum": ["direct", "summary"]},
                "freshness_intent": {"type": "boolean"},
                "query_language": {"type": "string", "minLength": 1},
                "search_query": {"type": "string"},
                "optimize_query": {"type": "boolean"},
                "search_mode": {"type": "string", "enum": ["auto", "deep"]},
                "max_results": {"type": "integer", "minimum": 1},
                "additional_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": _MAX_ADDITIONAL_QUERIES,
                },
            },
        }

        messages = self._build_messages(
            query=query,
            now_utc=now_utc,
            max_results_cap=max_results_cap,
        )
        result = await self._llm.chat(
            model=str(self.settings.answer.plan.use_model),
            messages=messages,
            schema=schema,
        )
        raw = (
            result.data
            if result.data is not None
            else _try_parse_json_value(result.text)
        )
        return self._normalize_plan(
            raw=raw,
            original_query=query,
            max_results_cap=max_results_cap,
        )

    def _build_messages(
        self,
        *,
        query: str,
        now_utc: datetime,
        max_results_cap: int,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a planner for a web-answer pipeline. Return JSON only. "
                    "You must decide question type, freshness intent, and search intensity "
                    "from the user question itself."
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
                        "- answer_mode: direct for concrete factual questions; summary for open-ended questions.",
                        "- freshness_intent: true ONLY when the question explicitly requests recency/currentness (e.g., latest/current/today/now/as of/this year/month, or explicit date anchor).",
                        "- do NOT infer freshness_intent=true for broad status/trend questions unless recency is explicit.",
                        "- only use Current UTC timestamp/date when freshness_intent=true.",
                        "- if freshness_intent=false, do not add date/time constraints into search_query or additional_queries.",
                        "- choose search intensity to minimize latency while preserving answer quality.",
                        "- max_results must be in [1, " + str(max_results_cap) + "].",
                        "- additional_queries must be empty when search_mode=auto.",
                        "- when search_mode=deep, populate additional_queries with 2-4 semantically distinct query variants to improve deep retrieval coverage.",
                        "- if freshness_intent=true, include explicit time constraints in search_query and additional_queries as needed.",
                        "Output fields:",
                        "- answer_mode: direct|summary",
                        "- freshness_intent: boolean",
                        "- query_language: language+script to match final answer (examples: English, Simplified Chinese, Japanese, Spanish). Determine this from Question only, not from search results.",
                        "- search_query: optimized query",
                        "- optimize_query: boolean",
                        "- search_mode: auto|deep",
                        "- max_results: integer",
                        "- additional_queries: string[] (<=4)",
                    ]
                ),
            },
        ]

    def _normalize_plan(
        self,
        *,
        raw: object,
        original_query: str,
        max_results_cap: int,
    ) -> _PlannedSearch:
        data = raw if isinstance(raw, dict) else {}

        raw_mode = clean_whitespace(str(data.get("answer_mode") or "")).casefold()
        answer_mode: _ANSWER_MODE = "direct" if raw_mode == "direct" else "summary"

        freshness_intent = _coerce_bool(data.get("freshness_intent"), default=False)
        query_language = clean_whitespace(str(data.get("query_language") or ""))
        if not query_language:
            query_language = "same as query"
        optimize_query = _coerce_bool(data.get("optimize_query"), default=False)
        planned_query = clean_whitespace(str(data.get("search_query") or ""))
        search_query = (
            planned_query
            if optimize_query and planned_query
            else clean_whitespace(original_query)
        )

        raw_mode = clean_whitespace(str(data.get("search_mode") or "")).casefold()
        search_mode: Literal["auto", "deep"] = "deep" if raw_mode == "deep" else "auto"

        default_max_results = min(max_results_cap, 5)
        raw_max_results = _coerce_int(
            data.get("max_results"),
            default=default_max_results,
        )
        max_results = max(1, min(max_results_cap, int(raw_max_results)))

        additional_queries: list[str] | None = None
        if search_mode == "deep" and isinstance(data.get("additional_queries"), list):
            additional_queries = _normalize_query_list(
                data.get("additional_queries") or [],
                limit=_MAX_ADDITIONAL_QUERIES,
            )

        return _PlannedSearch(
            answer_mode=answer_mode,
            freshness_intent=freshness_intent,
            query_language=query_language,
            search_query=search_query,
            search_mode=search_mode,
            max_results=max_results,
            additional_queries=additional_queries,
        )


def _normalize_query_list(raw_items: list[object], *, limit: int) -> list[str] | None:
    if limit <= 0:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        value = clean_whitespace(str(item or ""))
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= limit:
            break
    return out or None


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


def _coerce_int(value: object, *, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return int(default)


def _try_parse_json_value(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start : end + 1])
        raise


__all__ = ["AnswerPlanStep"]
