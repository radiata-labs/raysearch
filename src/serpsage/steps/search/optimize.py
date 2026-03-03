from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

from serpsage.models.pipeline import SearchStepContext
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


@dataclass(slots=True)
class _OptimizedQuery:
    search_query: str
    optimize_query: bool
    freshness_intent: bool
    query_language: str


class SearchOptimizeStep(StepBase[SearchStepContext]):
    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        mode = self._normalize_mode(ctx.request.mode)
        disable_internal_llm = bool(ctx.disable_internal_llm)
        optimize_enabled = mode in {"auto", "deep"}
        if disable_internal_llm:
            return ctx
        if not optimize_enabled:
            return ctx
        query = clean_whitespace(str(ctx.request.query or ""))
        if not query:
            return ctx
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        try:
            optimized = await self._optimize_query(
                query=query, now_utc=now_utc, mode=mode
            )
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="search.optimize.error",
                request_id=ctx.request_id,
                stage="search_optimize",
                status="error",
                error_code="search_query_optimize_failed",
                error_type=type(exc).__name__,
                attrs={
                    "request_id": ctx.request_id,
                    "mode": mode,
                    "query": query,
                    "message": str(exc),
                },
            )
            return ctx
        optimized_query = (
            optimized.search_query
            if optimized.optimize_query and optimized.search_query
            else query
        )
        query_changed = optimized_query != query
        if query_changed:
            ctx.request = ctx.request.model_copy(update={"query": optimized_query})
        return ctx

    async def _optimize_query(
        self,
        *,
        query: str,
        now_utc: datetime,
        mode: Literal["fast", "auto", "deep"],
    ) -> _OptimizedQuery:
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "search_query",
                "optimize_query",
                "freshness_intent",
                "query_language",
            ],
            "properties": {
                "search_query": {"type": "string"},
                "optimize_query": {"type": "boolean"},
                "freshness_intent": {"type": "boolean"},
                "query_language": {"type": "string", "minLength": 1},
            },
        }
        messages = self._build_messages(query=query, now_utc=now_utc, mode=mode)
        result = await self._llm.create(
            model=str(self.settings.answer.plan.use_model),
            messages=messages,
            response_format=schema,
        )
        raw = (
            result.data
            if result.data is not None
            else self._try_parse_json_value(result.text)
        )
        return self._normalize_output(raw=raw, original_query=query)

    def _build_messages(
        self,
        *,
        query: str,
        now_utc: datetime,
        mode: Literal["fast", "auto", "deep"],
    ) -> list[dict[str, str]]:
        mode_guidance = self._build_mode_guidance(mode=mode)
        return [
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer for a web retrieval pipeline. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        f"Current UTC timestamp: {now_utc.isoformat()}",
                        f"Current UTC date: {now_utc.date().isoformat()}",
                        f"Question: {query}",
                        f"Search mode: {mode}",
                        "Decision policy:",
                        "- freshness_intent: true ONLY when the question explicitly requests recency/currentness (e.g., latest/current/today/now/as of/this year/month, or explicit date anchor).",
                        "- do NOT infer freshness_intent=true for broad status/trend questions unless recency is explicit.",
                        "- only use Current UTC timestamp/date when freshness_intent=true.",
                        "- if freshness_intent=false, do not add date/time constraints into search_query.",
                        "- if freshness_intent=true, include explicit time constraints in search_query as needed.",
                        "- keep same language/script as the question.",
                        "- preserve core entities and intent; keep query concise and web-search friendly.",
                        "Mode guidance:",
                        *mode_guidance,
                        "Output fields:",
                        "- search_query: optimized query",
                        "- optimize_query: boolean",
                        "- freshness_intent: boolean",
                        "- query_language: language+script",
                    ]
                ),
            },
        ]

    def _build_mode_guidance(
        self, *, mode: Literal["fast", "auto", "deep"]
    ) -> list[str]:
        if mode == "deep":
            return [
                "- prioritize recall while keeping the same entities and intent boundaries.",
                "- include closely related facets only when they are likely to improve retrieval coverage.",
                "- avoid unrelated broadening or introducing new core entities.",
            ]
        return [
            "- prioritize precision and avoid semantic drift.",
            "- keep only constraints and qualifiers directly required by the original question.",
            "- do not broaden scope beyond the user's explicit intent.",
        ]

    def _normalize_output(self, *, raw: object, original_query: str) -> _OptimizedQuery:
        data = raw if isinstance(raw, dict) else {}
        optimize_query = _coerce_bool(data.get("optimize_query"), default=True)
        search_query = clean_whitespace(str(data.get("search_query") or ""))
        if not search_query:
            search_query = clean_whitespace(original_query)
        freshness_intent = _coerce_bool(data.get("freshness_intent"), default=False)
        query_language = clean_whitespace(str(data.get("query_language") or ""))
        if not query_language:
            query_language = "same as query"
        return _OptimizedQuery(
            search_query=search_query,
            optimize_query=optimize_query,
            freshness_intent=freshness_intent,
            query_language=query_language,
        )

    def _normalize_mode(self, value: object) -> Literal["fast", "auto", "deep"]:
        token = clean_whitespace(str(value or "")).casefold()
        if token in {"fast", "auto", "deep"}:
            return token  # type: ignore[return-value]
        return "auto"

    def _try_parse_json_value(self, text: str) -> object:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from text, but validate the extracted substring
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                candidate = text[start : end + 1]
                # Validate minimum structure before parsing
                if candidate.count("{") == candidate.count("}"):
                    return json.loads(candidate)
            raise


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


__all__ = ["SearchOptimizeStep"]
