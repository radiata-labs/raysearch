from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from pydantic import Field

from serpsage.core.model_base import MutableModel
from serpsage.models.pipeline import (
    ResearchLinkCandidate,
    ResearchRoundState,
    ResearchSearchJob,
    ResearchStepContext,
)
from serpsage.models.research import (
    ContentOutputPayload,
    OverviewOutputPayload,
    PlanOutputPayload,
    PlanSearchJobPayload,
    RoundAction,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.language import (
    language_alignment_score,
    normalize_language_code,
)
from serpsage.steps.research.prompt import (
    build_plan_messages as build_plan_prompt_messages,
)
from serpsage.steps.research.prompt import (
    build_query_language_repair_messages,
    render_link_candidates_markdown,
    render_queries_markdown,
    render_rounds_markdown,
    render_theme_plan_markdown,
)
from serpsage.steps.research.schema import (
    build_plan_schema,
    build_query_language_repair_schema,
)
from serpsage.steps.research.utils import (
    merge_strings,
    normalize_strings,
    resolve_research_model,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class _QueryLanguageRepairJobPayload(MutableModel):
    query: str = ""
    additional_queries: list[str] = Field(default_factory=list, max_length=8)


class _QueryLanguageRepairOutputPayload(MutableModel):
    search_jobs: list[_QueryLanguageRepairJobPayload] = Field(
        default_factory=list,
        max_length=8,
    )


class ResearchPlanStep(StepBase[ResearchStepContext]):
    _QUERY_LANGUAGE_ALIGNMENT_THRESHOLD = 0.65
    _LOW_GAIN_THRESHOLD = 0.05

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.runtime.stop:
            return ctx
        budget = ctx.runtime.budget
        if ctx.runtime.round_index >= budget.max_rounds:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_rounds"
            return ctx
        remain_fetch_calls = budget.max_fetch_calls - ctx.runtime.fetch_calls
        if remain_fetch_calls <= 0:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_fetch_calls"
            return ctx
        remain_search_calls = budget.max_search_calls - ctx.runtime.search_calls
        next_round_index = ctx.runtime.round_index + 1
        if remain_search_calls <= 0 and not self._can_attempt_explore(
            ctx=ctx,
            round_index=next_round_index,
        ):
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_search_calls"
            return ctx
        round_index = next_round_index
        ctx.runtime.round_index = round_index
        ctx.current_round = ResearchRoundState(round_index=round_index)
        ctx.work.search_jobs = []
        ctx.work.round_action = "search"
        ctx.work.explore_target_source_ids = []
        ctx.work.search_fetched_candidates = []
        ctx.work.overview_review = OverviewOutputPayload()
        ctx.work.content_review = ContentOutputPayload()
        ctx.work.need_content_source_ids = []
        ctx.work.next_queries = []
        core_question = self._resolve_core_question(ctx)
        if not core_question:
            core_question = ctx.request.themes
        candidate_queries = merge_strings(
            list(ctx.plan.next_queries),
            [core_question],
            limit=max(8, budget.max_queries_per_round * 3),
        )
        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        payload = PlanOutputPayload(
            query_strategy="mixed",
            round_action="search",
            explore_target_source_ids=[],
            search_jobs=[],
        )
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=self._build_plan_messages(
                    ctx=ctx,
                    candidate_queries=candidate_queries,
                    core_question=core_question,
                    now_utc=now_utc,
                ),
                response_format=PlanOutputPayload,
                format_override=build_plan_schema(),
                retries=self.settings.research.llm_self_heal_retries,
            )
            payload = chat_result.data
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.plan.error",
                request_id=ctx.request_id,
                stage="plan",
                status="error",
                error_code="research_round_plan_failed",
                error_type=type(exc).__name__,
                attrs={
                    "round_index": round_index,
                    "message": str(exc),
                },
            )
        strategy = clean_whitespace(payload.query_strategy or "mixed")
        allow_explore = self._can_attempt_explore(
            ctx=ctx,
            round_index=round_index,
        )
        round_action = self._normalize_round_action(
            raw=payload.round_action,
            round_index=round_index,
            allow_explore=allow_explore,
        )
        if remain_search_calls <= 0 and allow_explore:
            round_action = "explore"
        last_round_candidates = self._resolve_last_round_candidates(
            ctx=ctx,
            round_index=round_index,
        )
        explore_target_source_ids = self._normalize_explore_target_source_ids(
            raw=payload.explore_target_source_ids,
            candidates=last_round_candidates,
            limit=ctx.runtime.mode_depth.explore_target_pages_per_round,
        )
        if round_action == "explore" and not explore_target_source_ids:
            explore_target_source_ids = self._fallback_explore_target_source_ids(
                candidates=last_round_candidates,
                limit=ctx.runtime.mode_depth.explore_target_pages_per_round,
            )
        if round_action == "explore" and not explore_target_source_ids:
            round_action = "search"
        ctx.work.round_action = round_action
        ctx.work.explore_target_source_ids = list(explore_target_source_ids)
        ctx.current_round.query_strategy = strategy or round_action
        remain_search_calls = budget.max_search_calls - ctx.runtime.search_calls
        job_limit = max(0, min(budget.max_queries_per_round, remain_search_calls))
        jobs = self._normalize_jobs(
            payload.search_jobs,
            job_limit=job_limit,
            core_question=core_question,
            fallback_queries=candidate_queries,
        )
        jobs = self._enforce_low_budget_deep_policy(
            jobs=jobs,
            candidate_queries=candidate_queries,
            job_limit=job_limit,
        )
        search_language = self._resolve_search_language(ctx)
        (
            jobs,
            query_language_repair_applied,
        ) = await self._repair_jobs_language_if_needed(
            ctx=ctx,
            jobs=jobs,
            search_language=search_language,
            model=model,
            now_utc=now_utc,
        )
        if query_language_repair_applied:
            ctx.runtime.query_language_repair_applied = True
            await self.emit_tracking_event(
                event_name="research.search_language.query_repair_applied",
                request_id=ctx.request_id,
                stage="plan",
                attrs={
                    "round_index": round_index,
                    "search_language": search_language,
                    "jobs": len(jobs),
                },
            )
        jobs, fallback_applied = self._apply_progressive_language_fallback(
            ctx=ctx,
            jobs=jobs,
            job_limit=job_limit,
            round_action=round_action,
            search_language=search_language,
        )
        if fallback_applied:
            ctx.runtime.search_language_fallback_applied = True
            await self.emit_tracking_event(
                event_name="research.search_language.fallback_applied",
                request_id=ctx.request_id,
                stage="plan",
                attrs={
                    "round_index": round_index,
                    "search_language": search_language,
                    "output_language": normalize_language_code(
                        self._resolve_output_language(ctx)
                        or ctx.plan.theme_plan.input_language,
                        default="other",
                    ),
                },
            )
        if round_action == "search" and not jobs:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "no_queries"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "no_queries"
            ctx.rounds.append(ctx.current_round)
            return ctx
        ctx.work.search_jobs = jobs
        ctx.current_round.queries = [job.query for job in jobs]
        return ctx

    def _normalize_jobs(
        self,
        raw: list[PlanSearchJobPayload],
        *,
        job_limit: int,
        core_question: str,
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
            include_domains = normalize_strings(item.include_domains, limit=12)
            exclude_domains = normalize_strings(item.exclude_domains, limit=12)
            include_text = normalize_strings(item.include_text, limit=8)
            exclude_text = normalize_strings(item.exclude_text, limit=8)
            additional_queries = normalize_strings(
                item.additional_queries,
                limit=8,
            )
            if mode != "deep":
                additional_queries = []
            out.append(
                ResearchSearchJob(
                    query=query,
                    intent=intent,
                    mode=mode,  # type: ignore[arg-type]
                    additional_queries=additional_queries,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    include_text=include_text,
                    exclude_text=exclude_text,
                )
            )
            if len(out) >= job_limit:
                break
        if out:
            return out
        fallback = merge_strings([core_question], fallback_queries, limit=job_limit)
        return [
            ResearchSearchJob(
                query=item,
                intent="coverage",
                mode="auto",
            )
            for item in fallback
        ]

    def _enforce_low_budget_deep_policy(
        self,
        *,
        jobs: list[ResearchSearchJob],
        candidate_queries: list[str],
        job_limit: int,
    ) -> list[ResearchSearchJob]:
        if job_limit != 1 or not jobs:
            return jobs
        head = jobs[0]
        extras = normalize_strings(
            merge_strings(
                list(head.additional_queries),
                list(candidate_queries),
                limit=8,
            ),
            limit=8,
        )
        extras = [item for item in extras if item.casefold() != head.query.casefold()]
        if not extras:
            return jobs
        jobs[0] = head.model_copy(
            update={
                "mode": "deep",
                "additional_queries": extras,
            }
        )
        return jobs

    def _resolve_search_language(self, ctx: ResearchStepContext) -> str:
        from_plan = normalize_language_code(
            ctx.plan.theme_plan.search_language,
            default="other",
        )
        if from_plan != "other":
            return from_plan
        from_output = normalize_language_code(
            self._resolve_output_language(ctx) or ctx.plan.theme_plan.input_language,
            default="other",
        )
        if from_output != "other":
            return from_output
        return "en"

    def _jobs_need_language_repair(
        self,
        *,
        jobs: list[ResearchSearchJob],
        search_language: str,
    ) -> bool:
        if not jobs:
            return False
        target = normalize_language_code(search_language, default="other")
        if target == "other":
            return False
        scores: list[float] = []
        for job in jobs:
            scores.append(
                language_alignment_score(
                    text=job.query,
                    target_language=target,
                )
            )
            scores.extend(
                language_alignment_score(
                    text=extra,
                    target_language=target,
                )
                for extra in list(job.additional_queries or [])
            )
        if not scores:
            return False
        low_count = sum(
            1
            for score in scores
            if score < float(self._QUERY_LANGUAGE_ALIGNMENT_THRESHOLD)
        )
        return low_count > (len(scores) // 2)

    async def _repair_jobs_language_if_needed(
        self,
        *,
        ctx: ResearchStepContext,
        jobs: list[ResearchSearchJob],
        search_language: str,
        model: str,
        now_utc: datetime,
    ) -> tuple[list[ResearchSearchJob], bool]:
        if not jobs:
            return jobs, False
        if not self._jobs_need_language_repair(
            jobs=jobs,
            search_language=search_language,
        ):
            return jobs, False
        payload = _QueryLanguageRepairOutputPayload(
            search_jobs=[
                _QueryLanguageRepairJobPayload(
                    query=item.query,
                    additional_queries=list(item.additional_queries),
                )
                for item in jobs
            ]
        )
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=self._build_query_language_repair_messages(
                    ctx=ctx,
                    jobs=jobs,
                    search_language=search_language,
                    now_utc=now_utc,
                ),
                response_format=_QueryLanguageRepairOutputPayload,
                format_override=build_query_language_repair_schema(),
                retries=self.settings.research.llm_self_heal_retries,
            )
            payload = chat_result.data
        except Exception:
            return jobs, False
        repaired = self._normalize_repaired_search_jobs(
            raw=payload,
            baseline=jobs,
        )
        return repaired, repaired != jobs

    def _build_query_language_repair_messages(
        self,
        *,
        ctx: ResearchStepContext,
        jobs: list[ResearchSearchJob],
        search_language: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        output_language = normalize_language_code(
            self._resolve_output_language(ctx) or ctx.plan.theme_plan.input_language,
            default="other",
        )
        current_jobs = [
            {
                "query": item.query,
                "mode": item.mode,
                "additional_queries": list(item.additional_queries),
                "intent": item.intent,
            }
            for item in jobs
        ]
        return build_query_language_repair_messages(
            current_utc_date=now_utc.date().isoformat(),
            core_question=self._resolve_core_question(ctx),
            required_search_language=search_language,
            required_output_language=output_language,
            current_search_jobs_json=json.dumps(current_jobs, ensure_ascii=False),
        )

    def _normalize_repaired_search_jobs(
        self,
        *,
        raw: _QueryLanguageRepairOutputPayload,
        baseline: list[ResearchSearchJob],
    ) -> list[ResearchSearchJob]:
        payload_jobs = list(raw.search_jobs or [])
        if not payload_jobs:
            return baseline
        out: list[ResearchSearchJob] = []
        for index, base in enumerate(baseline):
            if index >= len(payload_jobs):
                out.append(base.model_copy(deep=True))
                continue
            item = payload_jobs[index]
            query = clean_whitespace(item.query) or base.query
            additional_queries = normalize_strings(
                list(item.additional_queries),
                limit=8,
            )
            if base.mode != "deep":
                additional_queries = []
            out.append(
                base.model_copy(
                    update={
                        "query": query,
                        "additional_queries": additional_queries,
                    },
                    deep=True,
                )
            )
        return out

    def _apply_progressive_language_fallback(
        self,
        *,
        ctx: ResearchStepContext,
        jobs: list[ResearchSearchJob],
        job_limit: int,
        round_action: RoundAction,
        search_language: str,
    ) -> tuple[list[ResearchSearchJob], bool]:
        if round_action != "search":
            return jobs, False
        if job_limit <= 0:
            return jobs, False
        if not self._should_apply_progressive_language_fallback(
            ctx=ctx,
            search_language=search_language,
        ):
            return jobs, False
        rescue_query = self._build_output_language_rescue_query(ctx=ctx)
        if not rescue_query:
            return jobs, False
        out = [item.model_copy(deep=True) for item in jobs]
        rescue_key = rescue_query.casefold()
        if any(item.query.casefold() == rescue_key for item in out):
            return out, False
        rescue_job = ResearchSearchJob(
            query=rescue_query,
            intent="refresh",
            mode="auto",
        )
        if len(out) < job_limit:
            out.append(rescue_job)
            return out, True
        if not out:
            return [rescue_job], True
        out[-1] = rescue_job
        return out, True

    def _should_apply_progressive_language_fallback(
        self,
        *,
        ctx: ResearchStepContext,
        search_language: str,
    ) -> bool:
        output_language = normalize_language_code(
            self._resolve_output_language(ctx) or ctx.plan.theme_plan.input_language,
            default="other",
        )
        search_lang = normalize_language_code(search_language, default="other")
        if output_language == "other" or search_lang == "other":
            return False
        if output_language == search_lang:
            return False
        rounds = list(ctx.rounds)[-2:]
        if len(rounds) < 2:
            return False
        return all(self._is_low_gain_round(item) for item in rounds)

    def _is_low_gain_round(self, round_state: ResearchRoundState) -> bool:
        if round_state.result_count <= 0:
            return True
        return float(round_state.corpus_score_gain) < float(self._LOW_GAIN_THRESHOLD)

    def _build_output_language_rescue_query(self, *, ctx: ResearchStepContext) -> str:
        query = self._resolve_core_question(ctx)
        if query:
            return query
        return ctx.request.themes

    def _build_plan_messages(
        self,
        *,
        ctx: ResearchStepContext,
        candidate_queries: list[str],
        core_question: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        budget = ctx.runtime.budget
        mode_depth = ctx.runtime.mode_depth
        out_lang = self._resolve_output_language(ctx) or "en"
        out_lang_name = out_lang or "unspecified"
        search_language = self._resolve_search_language(ctx)
        report_style = ctx.plan.theme_plan.report_style
        theme_plan_markdown = render_theme_plan_markdown(ctx.plan.theme_plan)
        previous_rounds_markdown = render_rounds_markdown(ctx.rounds, limit=3)
        candidate_queries_markdown = render_queries_markdown(candidate_queries)
        round_index = ctx.runtime.round_index
        last_round_candidates = self._resolve_last_round_candidates(
            ctx=ctx,
            round_index=round_index,
        )
        last_round_candidates_markdown = render_link_candidates_markdown(
            last_round_candidates,
            max_pages=max(1, mode_depth.explore_target_pages_per_round),
            max_links_per_page=6,
        )
        return build_plan_prompt_messages(
            theme=ctx.request.themes,
            core_question=core_question,
            report_style=report_style,
            mode_depth_profile=mode_depth.mode_key,
            round_index=round_index,
            current_utc_timestamp=now_utc.isoformat(),
            current_utc_date=now_utc.date().isoformat(),
            required_search_language=search_language,
            required_output_language=out_lang,
            required_output_language_label=out_lang_name,
            theme_plan_markdown=theme_plan_markdown,
            previous_rounds_markdown=previous_rounds_markdown,
            candidate_queries_markdown=candidate_queries_markdown,
            required_entities=list(ctx.plan.theme_plan.required_entities),
            search_calls_remaining=max(
                0,
                budget.max_search_calls - ctx.runtime.search_calls,
            ),
            fetch_calls_remaining=max(
                0,
                budget.max_fetch_calls - ctx.runtime.fetch_calls,
            ),
            max_queries_this_round=budget.max_queries_per_round,
            allow_explore=self._can_attempt_explore(
                ctx=ctx,
                round_index=round_index,
            ),
            explore_target_pages_per_round=mode_depth.explore_target_pages_per_round,
            explore_links_per_page=mode_depth.explore_links_per_page,
            last_round_link_candidates_markdown=last_round_candidates_markdown,
        )

    def _can_attempt_explore(
        self, *, ctx: ResearchStepContext, round_index: int
    ) -> bool:
        if round_index <= 1:
            return False
        if ctx.runtime.budget.max_fetch_calls <= ctx.runtime.fetch_calls:
            return False
        candidates = self._resolve_last_round_candidates(
            ctx=ctx,
            round_index=round_index,
        )
        return len(candidates) > 0

    def _resolve_last_round_candidates(
        self,
        *,
        ctx: ResearchStepContext,
        round_index: int,
    ) -> list[ResearchLinkCandidate]:
        expected_round = max(0, round_index - 1)
        if ctx.plan.last_round_link_candidates_round != expected_round:
            return []
        return [
            item.model_copy(deep=True) for item in ctx.plan.last_round_link_candidates
        ]

    def _normalize_round_action(
        self,
        *,
        raw: RoundAction | None,
        round_index: int,
        allow_explore: bool,
    ) -> RoundAction:
        if round_index <= 1:
            return "search"
        action_key = clean_whitespace(raw or "search").casefold()
        if action_key != "explore":
            return "search"
        return "explore" if allow_explore else "search"

    def _normalize_explore_target_source_ids(
        self,
        *,
        raw: list[int],
        candidates: list[ResearchLinkCandidate],
        limit: int,
    ) -> list[int]:
        if not raw or not candidates:
            return []
        cap = max(1, limit)
        allowed = {item.source_id for item in candidates}
        out: list[int] = []
        seen: set[int] = set()
        for item in raw:
            source_id = item
            if source_id in seen or source_id not in allowed:
                continue
            seen.add(source_id)
            out.append(source_id)
            if len(out) >= cap:
                break
        return out

    def _fallback_explore_target_source_ids(
        self,
        *,
        candidates: list[ResearchLinkCandidate],
        limit: int,
    ) -> list[int]:
        if not candidates:
            return []
        cap = max(1, limit)
        out: list[int] = []
        seen: set[int] = set()
        for item in candidates:
            source_id = item.source_id
            if source_id in seen:
                continue
            seen.add(source_id)
            out.append(source_id)
            if len(out) >= cap:
                break
        return out

    def _resolve_core_question(self, ctx: ResearchStepContext) -> str:
        return ctx.plan.theme_plan.core_question or ctx.request.themes

    def _resolve_output_language(self, ctx: ResearchStepContext) -> str:
        return ctx.plan.theme_plan.output_language


__all__ = ["ResearchPlanStep"]
