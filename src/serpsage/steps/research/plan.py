from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.pipeline import (
    ResearchRoundState,
    ResearchSearchJob,
    ResearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    add_error,
    chat_json,
    language_name,
    merge_strings,
    normalize_strings,
    resolve_research_model,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


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
        ctx.work.search_results = []
        ctx.work.abstract_review = {}
        ctx.work.content_review = {}
        ctx.work.need_content_source_ids = []
        ctx.work.next_queries = []

        candidate_queries = merge_strings(
            list(ctx.plan.next_queries),
            [ctx.request.themes],
            limit=max(8, int(budget.max_queries_per_round) * 3),
        )

        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["query_strategy", "search_jobs"],
            "properties": {
                "query_strategy": {
                    "type": "string",
                    "enum": ["coverage", "deepen", "verify", "refresh", "mixed"],
                },
                "search_jobs": {
                    "type": "array",
                    "maxItems": int(budget.max_queries_per_round),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query", "intent", "mode", "expected_gain"],
                        "properties": {
                            "query": {"type": "string"},
                            "intent": {
                                "type": "string",
                                "enum": ["coverage", "deepen", "verify", "refresh"],
                            },
                            "mode": {"type": "string", "enum": ["auto", "deep"]},
                            "include_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 8,
                            },
                            "exclude_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 8,
                            },
                            "include_text": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 8,
                            },
                            "exclude_text": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 8,
                            },
                            "expected_gain": {"type": "string"},
                        },
                    },
                },
            },
        }
        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        payload: dict[str, Any] = {}
        try:
            payload = await chat_json(
                llm=self._llm,
                model=model,
                messages=_build_plan_messages(ctx=ctx, candidate_queries=candidate_queries),
                schema=schema,
                retries=int(self.settings.research.llm_self_heal_retries),
            )
        except Exception as exc:  # noqa: BLE001
            add_error(
                ctx,
                code="research_round_plan_failed",
                message=str(exc),
                details={"round_index": round_index},
            )

        strategy = clean_whitespace(str(payload.get("query_strategy") or "mixed"))
        ctx.current_round.query_strategy = strategy or "mixed"
        remain_search_calls = int(budget.max_search_calls) - int(ctx.runtime.search_calls)
        job_limit = max(0, min(int(budget.max_queries_per_round), int(remain_search_calls)))
        raw_jobs = payload.get("search_jobs")
        jobs = self._normalize_jobs(raw_jobs, job_limit=job_limit, fallback_queries=candidate_queries)

        if not jobs:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "no_queries"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "no_queries"
            ctx.rounds.append(ctx.current_round)
            span.set_attr("stopped", True)
            span.set_attr("reason", "no_queries")
            return ctx

        ctx.work.search_jobs = jobs
        ctx.current_round.queries = [job.query for job in jobs]
        ctx.current_round.search_job_count = int(len(jobs))
        span.set_attr("round_index", int(round_index))
        span.set_attr("strategy", str(ctx.current_round.query_strategy))
        span.set_attr("search_jobs", int(len(jobs)))
        return ctx

    def _normalize_jobs(
        self,
        raw: object,
        *,
        job_limit: int,
        fallback_queries: list[str],
    ) -> list[ResearchSearchJob]:
        if job_limit <= 0:
            return []

        out: list[ResearchSearchJob] = []
        seen: set[str] = set()
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                query = clean_whitespace(str(item.get("query") or ""))
                if not query:
                    continue
                key = query.casefold()
                if key in seen:
                    continue
                seen.add(key)
                intent = clean_whitespace(str(item.get("intent") or "coverage")).casefold()
                if intent not in {"coverage", "deepen", "verify", "refresh"}:
                    intent = "coverage"
                mode = clean_whitespace(str(item.get("mode") or "auto")).casefold()
                if mode not in {"auto", "deep"}:
                    mode = "auto"
                out.append(
                    ResearchSearchJob(
                        query=query,
                        intent=intent,
                        mode=mode,  # type: ignore[arg-type]
                        include_domains=normalize_strings(item.get("include_domains"), limit=8),
                        exclude_domains=normalize_strings(item.get("exclude_domains"), limit=8),
                        include_text=normalize_strings(item.get("include_text"), limit=8),
                        exclude_text=normalize_strings(item.get("exclude_text"), limit=8),
                        expected_gain=clean_whitespace(str(item.get("expected_gain") or "")),
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
    *,
    ctx: ResearchStepContext,
    candidate_queries: list[str],
) -> list[dict[str, str]]:
    budget = ctx.runtime.budget
    out_lang = ctx.plan.output_language or "en"
    out_lang_name = language_name(out_lang)
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
                "1) Minimize overlap between jobs while maximizing information gain.\n"
                "2) Respect remaining budget and prioritize unresolved evidence gaps.\n"
                "3) Use deep mode only when higher recall or conflict verification is needed.\n"
                "4) Free-text fields must be in the required output language.\n"
                "5) Return JSON only; no markdown or explanations.\n"
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
                f"ROUND_INDEX:\n{ctx.runtime.round_index}\n\n"
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


__all__ = ["ResearchPlanStep"]
