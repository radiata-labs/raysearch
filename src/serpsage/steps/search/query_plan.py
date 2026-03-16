from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from serpsage.dependencies import Depends
from serpsage.models.steps.search import (
    QuerySourceSpec,
    SearchQueryJob,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from collections.abc import Sequence


class SearchQueryPlanStep(StepBase[SearchStepContext]):
    llm: LLMClientBase = Depends()
    provider: SearchProviderBase = Depends()

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        ctx.plan.aborted = False
        ctx.plan.abort_reason = ""
        ctx.plan.query_jobs = []

        primary_query = await self._build_primary_query(ctx)
        if primary_query is None:
            await self._abort_empty_query(ctx)
            return ctx

        ctx.plan.primary_query = primary_query
        if primary_query.query != ctx.request.query:
            ctx.request = ctx.request.model_copy(update={"query": primary_query.query})
        ctx.plan.query_jobs = await self._build_query_jobs(
            ctx=ctx,
            primary_query=primary_query,
        )
        await self.tracker.info(
            name="search.query_plan.completed",
            request_id=ctx.request_id,
            step="search.query_plan",
            data={
                "aborted": bool(ctx.plan.aborted),
                "query_count": len(ctx.plan.query_jobs or []),
                "primary_query": primary_query.query,
            },
        )
        await self.tracker.debug(
            name="search.query_plan.detail",
            request_id=ctx.request_id,
            step="search.query_plan",
            data={
                "queries": [
                    {
                        "query": job.query.query,
                        "source": job.source,
                        "include_sources": list(job.query.include_sources or []),
                    }
                    for job in (ctx.plan.query_jobs or [])
                ],
                "abort_reason": ctx.plan.abort_reason or "",
            },
        )
        return ctx

    async def _build_primary_query(
        self, ctx: SearchStepContext
    ) -> QuerySourceSpec | None:
        query = clean_whitespace(ctx.request.query)
        if not query:
            return None
        primary_query = QuerySourceSpec(query=query)
        if ctx.runtime.disable_internal_llm or ctx.plan.mode == "fast":
            return primary_query
        try:
            planned_query = await self._plan_primary_query(
                query=query,
                request_id=ctx.request_id,
                now_utc=datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC),
                mode=ctx.plan.mode,
                subsystem=ctx.runtime.engine_selection_subsystem or "search",
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="search.query_plan.failed",
                request_id=ctx.request_id,
                step="search.query_plan",
                error_code="search_primary_query_plan_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "phase": "primary_query",
                    "mode": ctx.plan.mode,
                    "query": query,
                },
            )
            return primary_query
        return planned_query or primary_query

    async def _build_query_jobs(
        self,
        *,
        ctx: SearchStepContext,
        primary_query: QuerySourceSpec,
    ) -> list[SearchQueryJob]:
        max_extra_queries = ctx.plan.max_extra_queries
        if max_extra_queries <= 0:
            return [SearchQueryJob(query=primary_query, source="primary")]

        manual_queries = (
            self._resolve_manual_queries(ctx) if ctx.plan.mode == "deep" else []
        )
        llm_queries: list[QuerySourceSpec] = []
        if ctx.plan.mode != "fast" and not ctx.runtime.disable_internal_llm:
            llm_queries = await self._collect_llm_queries(
                ctx=ctx,
                query=primary_query.query,
                request_id=ctx.request_id,
                max_queries=max_extra_queries,
            )

        candidates = [SearchQueryJob(query=primary_query, source="primary")]
        candidates.extend(
            SearchQueryJob(query=query, source="manual") for query in manual_queries
        )
        candidates.extend(
            SearchQueryJob(query=query, source="llm") for query in llm_queries
        )
        jobs = self._dedupe_jobs(
            candidates=candidates,
            max_extra_queries=max_extra_queries,
        )
        if jobs:
            return jobs
        return [SearchQueryJob(query=primary_query, source="primary")]

    def _resolve_manual_queries(self, ctx: SearchStepContext) -> list[QuerySourceSpec]:
        values: Sequence[object] = ctx.runtime.additional_queries or [
            QuerySourceSpec(query=item)
            for item in (ctx.request.additional_queries or [])
        ]
        return self._normalize_queries(values)

    async def _plan_primary_query(
        self,
        *,
        query: str,
        request_id: str,
        now_utc: datetime,
        mode: Literal["fast", "auto", "deep"],
        subsystem: Literal["search", "research", "answer"],
    ) -> QuerySourceSpec | None:
        routes = resolve_engine_selection_routes(
            settings=self.settings,
            subsystem=subsystem,
            provider=self.provider,
        )
        engine_selection_context = build_engine_selection_context(routes=routes)
        response_format: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["query", "use_query"],
            "properties": {
                "query": self._build_primary_query_schema(has_routes=bool(routes)),
                "use_query": {"type": "boolean"},
            },
        }
        result = await self.llm.create(
            model=self.settings.answer.plan.use_model,
            messages=self._build_primary_query_messages(
                query=query,
                now_utc=now_utc,
                mode=mode,
                engine_selection_context=engine_selection_context,
            ),
            response_format=response_format,
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
        raw = (
            result.data
            if result.data is not None
            else self._try_parse_json_value(result.text)
        )
        if not isinstance(raw, dict):
            raise TypeError("query planning output must be an object")
        if not _coerce_bool(raw.get("use_query"), default=True):
            return None
        value = raw.get("query")
        if value is None:
            return None
        return QuerySourceSpec.model_validate(value)

    def _build_primary_query_schema(self, *, has_routes: bool) -> dict[str, Any]:
        if not has_routes:
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

    def _build_primary_query_messages(
        self,
        *,
        query: str,
        now_utc: datetime,
        mode: Literal["fast", "auto", "deep"],
        engine_selection_context: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You plan the primary search query for a web retrieval pipeline. "
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
                        "Requirements:",
                        "- keep the same language and script as the original question.",
                        "- preserve the original entities, scope, and intent.",
                        "- keep the query concise and web-search friendly.",
                        "- add time constraints only when the question explicitly asks for recency or a date anchor.",
                        (
                            "ENGINE_SELECTION_OUTPUT_RULES:\n"
                            "- query must be an object with query and include_sources.\n"
                            "- Prefer include_sources=[] for broad first-pass retrieval.\n"
                            "- Restrict engines only when the query clearly targets a specific evidence route.\n\n"
                            f"{engine_selection_context}"
                        )
                        if engine_selection_context
                        else "",
                        "Mode guidance:",
                        *self._build_mode_guidance(mode),
                        "Output fields:",
                        "- query: primary search query",
                        "- use_query: boolean",
                    ]
                ),
            },
        ]

    def _build_mode_guidance(self, mode: Literal["fast", "auto", "deep"]) -> list[str]:
        if mode == "deep":
            return [
                "- prioritize recall without introducing new core entities.",
                "- keep related qualifiers only when they improve retrieval coverage.",
            ]
        return [
            "- prioritize precision and avoid semantic drift.",
            "- keep only qualifiers required by the original question.",
        ]

    def _build_expansion_query_schema(self, *, has_routes: bool) -> dict[str, Any]:
        if not has_routes:
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

    def _build_expansion_query_messages(
        self,
        *,
        query: str,
        max_queries: int,
        engine_selection_context: str,
    ) -> list[dict[str, str]]:
        lines = [
            f"Primary query: {query}",
            f"Generate up to {max_queries} additional search queries.",
            "Requirements:",
            "- keep the same language and script as the primary query.",
            "- keep the same entities and intent boundaries.",
            "- keep each query concise and web-search friendly.",
            "- make each query semantically distinct from the others.",
            "- include time or entity qualifiers only when implied by the primary query.",
        ]
        if engine_selection_context:
            lines.append(
                "ENGINE_SELECTION_OUTPUT_RULES:\n"
                "- Each query in the array must be an object with query and include_sources.\n"
                "- Prefer include_sources=[] for broad first-pass retrieval.\n"
                "- Restrict engines only when the query clearly targets a specific evidence route.\n\n"
                f"{engine_selection_context}"
            )
        else:
            lines.append('Output JSON: {"queries": ["...", "..."]}')
        return [
            {
                "role": "system",
                "content": (
                    "You generate additional deep-search queries for a web retrieval pipeline. "
                    "Return JSON only."
                ),
            },
            {"role": "user", "content": "\n".join(lines)},
        ]

    async def _collect_llm_queries(
        self,
        *,
        ctx: SearchStepContext,
        query: str,
        request_id: str,
        max_queries: int,
    ) -> list[QuerySourceSpec]:
        model_name = self._resolve_expansion_model()
        try:
            return await self._expand_queries(
                query=query,
                request_id=request_id,
                model_name=model_name,
                max_queries=max_queries,
                subsystem=ctx.runtime.engine_selection_subsystem or "search",
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="search.query_plan.failed",
                request_id=ctx.request_id,
                step="search.query_plan",
                error_code="search_query_expansion_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "phase": "expand",
                    "query": query,
                    "model": model_name,
                },
            )
            return []

    def _resolve_expansion_model(self) -> str:
        model_name = clean_whitespace(self.settings.search.expansion.llm_model or "")
        return model_name or self.settings.answer.plan.use_model

    async def _expand_queries(
        self,
        *,
        query: str,
        request_id: str,
        model_name: str,
        max_queries: int,
        subsystem: Literal["search", "research", "answer"],
    ) -> list[QuerySourceSpec]:
        routes = resolve_engine_selection_routes(
            settings=self.settings,
            subsystem=subsystem,
            provider=self.provider,
        )
        engine_selection_context = build_engine_selection_context(routes=routes)
        has_routes = bool(routes)

        response_format: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["queries"],
            "properties": {
                "queries": {
                    "type": "array",
                    "items": self._build_expansion_query_schema(has_routes=has_routes),
                    "maxItems": max_queries,
                }
            },
        }
        result = await self.llm.create(
            model=model_name,
            messages=self._build_expansion_query_messages(
                query=query,
                max_queries=max_queries,
                engine_selection_context=engine_selection_context,
            ),
            response_format=response_format,
            timeout_s=self.settings.search.expansion.llm_timeout_s,
        )
        await self.meter.record(
            name="llm.tokens",
            request_id=request_id,
            model=str(model_name),
            unit="token",
            tokens={
                "prompt_tokens": int(result.usage.prompt_tokens),
                "completion_tokens": int(result.usage.completion_tokens),
                "total_tokens": int(result.usage.total_tokens),
            },
        )
        raw = (
            result.data
            if result.data is not None
            else self._try_parse_json_value(result.text)
        )
        if not isinstance(raw, dict):
            raise TypeError("query planning output must be an object")
        values = raw.get("queries")
        if not isinstance(values, list):
            raise TypeError("query planning output must contain array field `queries`")
        return self._normalize_queries(values)[:max_queries]

    def _dedupe_jobs(
        self,
        *,
        candidates: list[SearchQueryJob],
        max_extra_queries: int,
    ) -> list[SearchQueryJob]:
        exact_seen: set[str] = set()
        token_signatures: set[tuple[str, ...]] = set()
        jobs: list[SearchQueryJob] = []
        additional_count = 0

        for candidate in candidates:
            query = clean_whitespace(candidate.query.query)
            if not query:
                continue
            key = query.casefold()
            if key in exact_seen:
                continue
            if candidate.source != "primary" and additional_count >= max_extra_queries:
                continue

            signature = self._build_token_signature(query)
            if signature and signature in token_signatures:
                continue

            exact_seen.add(key)
            if signature:
                token_signatures.add(signature)
            jobs.append(
                SearchQueryJob(
                    query=candidate.query.model_copy(
                        update={"query": query}, deep=True
                    ),
                    source=candidate.source,
                )
            )
            if candidate.source != "primary":
                additional_count += 1
        return jobs

    def _build_token_signature(self, query: str) -> tuple[str, ...]:
        return tuple(sorted({token for token in tokenize_for_query(query) if token}))

    def _normalize_queries(self, values: Sequence[object]) -> list[QuerySourceSpec]:
        queries: list[QuerySourceSpec] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for value in values:
            if isinstance(value, QuerySourceSpec):
                query = value.model_copy(deep=True)
            else:
                query = QuerySourceSpec.model_validate(value)
            key = (query.query.casefold(), tuple(query.include_sources))
            if key in seen:
                continue
            seen.add(key)
            queries.append(query)
        return queries

    async def _abort_empty_query(self, ctx: SearchStepContext) -> None:
        ctx.plan.aborted = True
        ctx.plan.abort_reason = "empty query"
        await self.tracker.error(
            name="search.query_plan.failed",
            request_id=ctx.request_id,
            step="search.query_plan",
            error_code="search_query_plan_failed",
            error_message="empty query",
            data={
                "phase": "validate",
                "query": ctx.request.query,
            },
        )

    def _try_parse_json_value(self, text: str) -> object:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                candidate = text[start : end + 1]
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


__all__ = ["SearchQueryPlanStep"]
