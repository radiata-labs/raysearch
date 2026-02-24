from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from pydantic import BaseModel, ConfigDict, Field

from serpsage.models.errors import AppError
from serpsage.models.pipeline import (
    ResearchRoundState,
    ResearchSearchJob,
    ResearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    chat_pydantic,
    merge_strings,
    normalize_strings,
    resolve_research_model,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class _PlanSearchJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    query: str
    intent: str = "coverage"
    mode: str = "auto"
    include_domains: list[str] = Field(default_factory=list, max_length=8)
    exclude_domains: list[str] = Field(default_factory=list, max_length=8)
    include_text: list[str] = Field(default_factory=list, max_length=8)
    exclude_text: list[str] = Field(default_factory=list, max_length=8)
    expected_gain: str = ""


class _PlanOutputPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    query_strategy: str = "mixed"
    search_jobs: list[_PlanSearchJobPayload] = Field(default_factory=list, max_length=8)


class ResearchPlanStep(StepBase[ResearchStepContext]):
    span_name = "step.research_plan"

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.runtime.stop:
            span.set_attr("skipped", True)
            return ctx

        budget = ctx.runtime.budget
        if ctx.runtime.round_index >= int(budget.max_rounds):
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_rounds"
            span.set_attr("stopped", True)
            span.set_attr("reason", "max_rounds")
            return ctx
        if ctx.runtime.search_calls >= int(budget.max_search_calls):
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_search_calls"
            span.set_attr("stopped", True)
            span.set_attr("reason", "max_search_calls")
            return ctx

        round_index = int(ctx.runtime.round_index) + 1
        ctx.runtime.round_index = round_index
        ctx.current_round = ResearchRoundState(round_index=round_index)
        ctx.work.search_jobs = []
        ctx.work.abstract_review = {}
        ctx.work.content_review = {}
        ctx.work.need_content_source_ids = []
        ctx.work.next_queries = []

        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)
        if not core_question:
            core_question = ctx.request.themes
        candidate_queries = merge_strings(
            list(ctx.plan.next_queries),
            [core_question],
            limit=max(8, int(budget.max_queries_per_round) * 3),
        )

        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        payload = _PlanOutputPayload(query_strategy="mixed", search_jobs=[])
        try:
            payload = await chat_pydantic(
                llm=self._llm,
                model=model,
                messages=self._build_plan_messages(
                    ctx=ctx,
                    candidate_queries=candidate_queries,
                    core_question=core_question,
                    now_utc=now_utc,
                ),
                schema_model=_PlanOutputPayload,
                retries=int(self.settings.research.llm_self_heal_retries),
                schema_json=self._build_plan_schema(),
            )
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="research_round_plan_failed",
                    message=str(exc),
                    details={"round_index": round_index},
                )
            )

        strategy = clean_whitespace(str(payload.query_strategy or "mixed"))
        ctx.current_round.query_strategy = strategy or "mixed"
        remain_search_calls = int(budget.max_search_calls) - int(
            ctx.runtime.search_calls
        )
        job_limit = max(
            0, min(int(budget.max_queries_per_round), int(remain_search_calls))
        )
        jobs = self._normalize_jobs(
            payload.search_jobs,
            job_limit=job_limit,
            fallback_queries=candidate_queries,
        )
        jobs = self._apply_core_question_guard(
            jobs=jobs,
            core_question=core_question,
            job_limit=job_limit,
        )

        if not jobs:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "no_queries"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "no_queries"
            ctx.rounds.append(ctx.current_round)
            print(
                "[research.plan]",
                json.dumps(
                    {
                        "round_index": int(round_index),
                        "core_question": core_question,
                        "candidate_queries": candidate_queries,
                        "query_strategy": str(ctx.current_round.query_strategy),
                        "raw_model_payload": payload.model_dump(),
                        "search_jobs": [],
                        "stop": True,
                        "stop_reason": "no_queries",
                    },
                    ensure_ascii=False,
                ),
            )
            span.set_attr("stopped", True)
            span.set_attr("reason", "no_queries")
            return ctx

        ctx.work.search_jobs = jobs
        ctx.current_round.queries = [job.query for job in jobs]
        print(
            "[research.plan]",
            json.dumps(
                {
                    "round_index": int(round_index),
                    "core_question": core_question,
                    "candidate_queries": candidate_queries,
                    "query_strategy": str(ctx.current_round.query_strategy),
                    "search_jobs": [job.model_dump() for job in jobs],
                    "raw_model_payload": payload.model_dump(),
                },
                ensure_ascii=False,
            ),
        )
        span.set_attr("round_index", int(round_index))
        span.set_attr("strategy", str(ctx.current_round.query_strategy))
        span.set_attr("search_jobs", int(len(jobs)))
        return ctx

    def _normalize_jobs(
        self,
        raw: list[_PlanSearchJobPayload],
        *,
        job_limit: int,
        fallback_queries: list[str],
    ) -> list[ResearchSearchJob]:
        if job_limit <= 0:
            return []

        out: list[ResearchSearchJob] = []
        seen: set[str] = set()
        for item in raw:
            query = clean_whitespace(item.query)
            if not query:
                continue
            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            intent = clean_whitespace(item.intent).casefold()
            if intent not in {"coverage", "deepen", "verify", "refresh"}:
                intent = "coverage"
            mode = clean_whitespace(item.mode).casefold()
            if mode not in {"auto", "deep"}:
                mode = "auto"
            out.append(
                ResearchSearchJob(
                    query=query,
                    intent=intent,
                    mode=mode,  # type: ignore[arg-type]
                    include_domains=normalize_strings(item.include_domains, limit=8),
                    exclude_domains=normalize_strings(item.exclude_domains, limit=8),
                    include_text=normalize_strings(item.include_text, limit=8),
                    exclude_text=normalize_strings(item.exclude_text, limit=8),
                    expected_gain=clean_whitespace(item.expected_gain),
                )
            )
            if len(out) >= job_limit:
                break

        if out:
            return out
        fallback = merge_strings(fallback_queries, [], limit=job_limit)
        return [
            ResearchSearchJob(
                query=item,
                intent="coverage",
                mode="auto",
                expected_gain="Increase coverage of core subthemes.",
            )
            for item in fallback
        ]

    def _build_plan_messages(
        self,
        *,
        ctx: ResearchStepContext,
        candidate_queries: list[str],
        core_question: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        budget = ctx.runtime.budget
        out_lang = ctx.plan.output_language or "en"
        out_lang_name = clean_whitespace(out_lang) or "unspecified"
        return [
            {
                "role": "system",
                "content": (
                    "Role: Principal Research Planner and Evidence Operations Lead.\n"
                    "Mission: Generate high-information, low-overlap search jobs for the next research round.\n"
                    "Instruction Priority:\n"
                    "P1) Schema correctness and budget adherence.\n"
                    "P2) Evidence gain per query.\n"
                    "P3) Output language consistency.\n"
                    "Hard Constraints:\n"
                    "1) All search_jobs must serve CORE_QUESTION. Do not introduce a new independent research question.\n"
                    "2) Minimize overlap between jobs while maximizing information gain.\n"
                    "3) Respect remaining budget and prioritize unresolved evidence gaps.\n"
                    "4) Use deep mode only when higher recall or conflict verification is needed.\n"
                    "5) Free-text fields must be in the required output language.\n"
                    "6) Temporal grounding: interpret relative time words against current UTC date.\n"
                    "7) If recency intent exists, include explicit temporal constraints in query text.\n"
                    "8) For high-impact claims, prioritize authoritative evidence routes (official documentation, primary sources, standards, vendor announcements, peer-reviewed or institution-backed reports).\n"
                    "9) Return JSON only; no markdown or explanations.\n"
                    "Allowed Evidence:\n"
                    "- User theme, theme plan, previous round summaries, candidate queries.\n"
                    "Failure Policy:\n"
                    "- If uncertain, produce fewer but higher-value jobs.\n"
                    "Quality Checklist:\n"
                    "- Distinct intent per job, explicit expected gain, no near duplicates, conflict-aware targeting."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"THEME:\n{ctx.request.themes}\n\n"
                    f"CORE_QUESTION:\n{core_question}\n\n"
                    f"ROUND_INDEX:\n{ctx.runtime.round_index}\n\n"
                    "TIME_CONTEXT:\n"
                    f"- current_utc_timestamp={now_utc.isoformat()}\n"
                    f"- current_utc_date={now_utc.date().isoformat()}\n\n"
                    "TEMPORAL_POLICY:\n"
                    "- If theme/round context asks for latest/current/today/now/as of/this year/month/week, queries must include concrete temporal anchors.\n"
                    "- When no recency intent is present, avoid over-constraining by date.\n"
                    "- Prefer fresh/authoritative sources for recency queries.\n\n"
                    "LANGUAGE_POLICY:\n"
                    f"- required_output_language={out_lang} ({out_lang_name})\n"
                    "- Keep textual fields in the required output language.\n\n"
                    f"THEME_PLAN:\n{ctx.plan.theme_plan}\n\n"
                    f"PREVIOUS_ROUNDS:\n{[r.model_dump() for r in ctx.rounds[-3:]]}\n\n"
                    f"CANDIDATE_QUERIES:\n{candidate_queries}\n\n"
                    "BUDGET_REMAINING:\n"
                    f"- search_calls_remaining={max(0, budget.max_search_calls - ctx.runtime.search_calls)}\n"
                    f"- fetch_calls_remaining={max(0, budget.max_fetch_calls - ctx.runtime.fetch_calls)}\n"
                    f"- max_queries_this_round={budget.max_queries_per_round}\n\n"
                    "Job design rubric:\n"
                    "- coverage: close missing subtheme coverage.\n"
                    "- deepen: improve depth on high-value evidence branches.\n"
                    "- verify: target contradiction resolution and tie-break evidence.\n"
                    "- refresh: prioritize latest authoritative updates."
                ),
            },
        ]

    def _build_plan_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["query_strategy", "search_jobs"],
            "properties": {
                "query_strategy": {"type": "string"},
                "search_jobs": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string"},
                            "intent": {"type": "string"},
                            "mode": {"type": "string"},
                            "include_domains": {
                                "type": "array",
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "exclude_domains": {
                                "type": "array",
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "include_text": {
                                "type": "array",
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "exclude_text": {
                                "type": "array",
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "expected_gain": {"type": "string"},
                        },
                    },
                },
            },
        }

    def _apply_core_question_guard(
        self,
        *,
        jobs: list[ResearchSearchJob],
        core_question: str,
        job_limit: int,
    ) -> list[ResearchSearchJob]:
        if not jobs:
            return []
        kept: list[ResearchSearchJob] = []
        rejected: list[str] = []
        for item in jobs:
            if self._is_query_focused(item.query, core_question):
                kept.append(item)
                if len(kept) >= max(1, int(job_limit)):
                    break
                continue
            rejected.append(item.query)

        if kept:
            return kept
        if job_limit <= 0:
            return []
        rewritten = self._rewrite_query_for_core_question(
            core_question=core_question,
            rejected_queries=rejected,
        )
        return [
            ResearchSearchJob(
                query=rewritten,
                intent="coverage",
                mode="auto",
                expected_gain="Recover focus on the core question after off-topic drift.",
            )
        ]

    def _rewrite_query_for_core_question(
        self,
        *,
        core_question: str,
        rejected_queries: list[str],
    ) -> str:
        core = clean_whitespace(core_question)
        if not core:
            return ""
        core_terms = self._extract_terms(core)
        aux_hint = ""
        for query in rejected_queries:
            for term in self._extract_terms(query):
                if term in core_terms or len(term) <= 1:
                    continue
                aux_hint = term
                break
            if aux_hint:
                break
        if not aux_hint:
            return core
        return clean_whitespace(f"{core} {aux_hint}")

    def _is_query_focused(self, query: str, core_question: str) -> bool:
        query_text = clean_whitespace(query).casefold()
        core_text = clean_whitespace(core_question).casefold()
        if not query_text or not core_text:
            return False
        if core_text in query_text or query_text in core_text:
            return True
        query_terms = self._extract_terms(query_text)
        core_terms = self._extract_terms(core_text)
        if not query_terms or not core_terms:
            return False
        overlap = len(query_terms & core_terms)
        core_ratio = overlap / max(1, len(core_terms))
        query_ratio = overlap / max(1, len(query_terms))
        cfg = self.settings.research.parallel
        return overlap >= int(cfg.focus_min_term_overlap) and (
            core_ratio >= float(cfg.focus_min_overlap_ratio)
            or query_ratio >= float(cfg.focus_min_overlap_ratio)
        )

    def _extract_terms(self, text: str) -> set[str]:
        cleaned = clean_whitespace(text).casefold()
        terms = {item for item in re.findall(r"[a-z0-9]+", cleaned) if item}
        for token in re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff]+", cleaned):
            if not token:
                continue
            terms.add(token)
            terms.update(ch for ch in token if ch.strip())
        return terms


__all__ = ["ResearchPlanStep"]
