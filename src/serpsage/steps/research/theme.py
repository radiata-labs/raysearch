from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import (
    ResearchModeDepthState,
    ResearchQuestionCard,
    ResearchStepContext,
)
from serpsage.models.research import (
    ReportStyle,
    ResearchThemePlan,
    ResearchThemePlanCard,
    TaskComplexity,
    TaskIntent,
    ThemeOutputPayload,
    ThemeQuestionCardPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.language import (
    detect_input_language,
    normalize_language_code,
    route_search_language,
)
from serpsage.steps.research.prompt import (
    build_theme_messages as build_theme_prompt_messages,
)
from serpsage.steps.research.prompt import (
    infer_report_style_from_theme,
    resolve_report_style,
)
from serpsage.steps.research.schema import build_theme_schema
from serpsage.steps.research.utils import (
    merge_strings,
    normalize_strings,
    resolve_research_model,
)

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class ResearchThemeStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        mode_depth = ctx.runtime.mode_depth
        prompt_card_cap = max(1, mode_depth.max_question_cards_effective)
        seed_limit = max(6, ctx.runtime.budget.max_queries_per_round * 3)
        fallback_report_style = infer_report_style_from_theme(ctx.request.themes)
        fallback_input_language = detect_input_language(ctx.request.themes)
        fallback_search_language = route_search_language(
            theme=ctx.request.themes,
            input_language=fallback_input_language,
        )
        fallback_task_intent = self._normalize_task_intent(
            raw=None,
            theme=ctx.request.themes,
            report_style=fallback_report_style,
        )
        fallback_complexity = self._normalize_task_complexity(
            raw=None,
            theme=ctx.request.themes,
            task_intent=fallback_task_intent,
        )
        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        payload = ThemeOutputPayload(
            detected_input_language=fallback_input_language,
            search_language=fallback_search_language,
            core_question=ctx.request.themes,
            report_style=fallback_report_style,
            task_intent=fallback_task_intent,
            complexity_tier=fallback_complexity,
            subthemes=[],
            required_entities=[],
            question_cards=[],
        )
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=self._build_theme_messages(
                    ctx,
                    now_utc=now_utc,
                    card_cap=prompt_card_cap,
                    report_style=fallback_report_style,
                ),
                response_format=ThemeOutputPayload,
                format_override=build_theme_schema(card_cap=prompt_card_cap),
                retries=self.settings.research.llm_self_heal_retries,
            )
            payload = chat_result.data
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.theme.error",
                request_id=ctx.request_id,
                stage="theme_plan",
                status="error",
                error_code="research_theme_plan_failed",
                error_type=type(exc).__name__,
                attrs={
                    "model": model,
                    "message": str(exc),
                },
            )
        input_language = normalize_language_code(
            payload.detected_input_language,
            default=fallback_input_language,
        )
        core_question = (payload.core_question or ctx.request.themes).strip()
        if not core_question:
            core_question = ctx.request.themes
        resolved_theme = core_question or ctx.request.themes
        search_language = normalize_language_code(
            payload.search_language,
            default="other",
        )
        if search_language == "other":
            search_language = route_search_language(
                theme=resolved_theme,
                input_language=input_language,
            )
        report_style = resolve_report_style(
            raw_style=payload.report_style,
            theme=resolved_theme,
        )
        raw_style_token = payload.report_style.strip().casefold()
        style_fallback_used = raw_style_token != report_style
        task_intent = self._normalize_task_intent(
            raw=payload.task_intent,
            theme=resolved_theme,
            report_style=report_style,
        )
        raw_task_intent_token = payload.task_intent.strip().casefold()
        intent_fallback_used = raw_task_intent_token != task_intent
        complexity_tier = self._normalize_task_complexity(
            raw=payload.complexity_tier,
            theme=resolved_theme,
            task_intent=task_intent,
        )
        raw_complexity_token = payload.complexity_tier.strip().casefold()
        complexity_fallback_used = raw_complexity_token != complexity_tier
        adaptive_applied = self._apply_mode_adaptive_depth(
            ctx=ctx,
            complexity_tier=complexity_tier,
        )
        mode_depth = ctx.runtime.mode_depth
        mode_key = mode_depth.mode_key
        card_cap = max(1, mode_depth.max_question_cards_effective)
        subthemes = normalize_strings(payload.subthemes, limit=12)
        required_entities = normalize_strings(payload.required_entities, limit=16)
        cards = self._normalize_question_cards(
            payload.question_cards,
            core_question=core_question,
            card_cap=card_cap,
            seed_limit=seed_limit,
            fallback_branches=merge_strings(
                subthemes,
                [core_question],
                limit=max(24, card_cap * 2),
            ),
        )
        ctx.plan.theme_plan.input_language = input_language
        ctx.plan.theme_plan.search_language = search_language
        ctx.plan.theme_plan.output_language = input_language
        ctx.plan.theme_plan.core_question = core_question
        ctx.parallel.question_cards = [item.model_copy(deep=True) for item in cards]
        theme_plan = ResearchThemePlan(
            core_question=core_question,
            report_style=report_style,
            task_intent=task_intent,
            complexity_tier=complexity_tier,
            subthemes=subthemes,
            required_entities=required_entities,
            input_language=input_language,
            search_language=search_language,
            output_language=input_language,
            question_cards=[
                ResearchThemePlanCard(
                    question_id=card.question_id,
                    question=card.question,
                    priority=card.priority,
                    seed_queries=list(card.seed_queries),
                    evidence_focus=list(card.evidence_focus),
                    expected_gain=card.expected_gain,
                )
                for card in cards
            ],
        )
        ctx.plan.theme_plan = theme_plan
        seed_groups = [list(item.seed_queries) for item in cards]
        next_query_limit = max(8, ctx.runtime.budget.max_queries_per_round * 3)
        ctx.plan.next_queries = merge_strings(
            *seed_groups,
            [core_question],
            limit=next_query_limit,
        )
        ctx.corpus.coverage_state.total_subthemes = len(subthemes)
        ctx.notes.append(
            f"Theme plan built with {len(cards)} question cards and {len(subthemes)} subthemes."
        )
        ctx.notes.append(f"Search language fixed to {theme_plan.search_language}.")
        ctx.notes.append(f"Report style fixed to `{report_style}`.")
        ctx.notes.append(
            f"Task intent fixed to `{task_intent}` with complexity tier `{complexity_tier}`."
        )
        if adaptive_applied:
            ctx.notes.append(
                f"Adaptive `{mode_key}` depth applied based on theme complexity."
            )
        if required_entities:
            ctx.notes.append(f"Required entities: {', '.join(required_entities[:8])}.")
        ctx.notes.append(f"Output language fixed to {theme_plan.output_language}.")
        await self.emit_tracking_event(
            event_name="research.language.selected",
            request_id=ctx.request_id,
            stage="theme_plan",
            attrs={
                "input_language": theme_plan.input_language,
                "output_language": theme_plan.output_language,
                "search_language": theme_plan.search_language,
            },
        )
        await self.emit_tracking_event(
            event_name="research.style.selected",
            request_id=ctx.request_id,
            stage="theme_plan",
            attrs={
                "report_style_selected": report_style,
                "report_style_fallback_used": style_fallback_used,
                "task_intent_selected": task_intent,
                "task_intent_fallback_used": intent_fallback_used,
                "complexity_tier_selected": complexity_tier,
                "complexity_tier_fallback_used": complexity_fallback_used,
            },
        )
        if mode_key in {"research", "research-pro"}:
            await self.emit_tracking_event(
                event_name="research.mode_depth.adaptive_applied",
                request_id=ctx.request_id,
                stage="theme_plan",
                attrs={
                    "mode_depth_profile": mode_key,
                    "task_intent": task_intent,
                    "complexity_tier": complexity_tier,
                    "effective_complexity_tier": complexity_tier,
                    "adaptive_applied": adaptive_applied,
                    "max_question_cards_effective": card_cap,
                    "min_rounds_per_track": mode_depth.min_rounds_per_track,
                    "gap_closure_passes": mode_depth.gap_closure_passes,
                    "density_gate_passes": mode_depth.density_gate_passes,
                },
            )
        await self.emit_tracking_event(
            event_name="research.theme.summary",
            request_id=ctx.request_id,
            stage="theme_plan",
            attrs={
                "core_question": theme_plan.core_question,
                "question_cards": len(cards),
                "subthemes": len(subthemes),
                "next_queries": len(ctx.plan.next_queries),
                "input_language": theme_plan.input_language,
                "search_language": theme_plan.search_language,
                "output_language": theme_plan.output_language,
                "mode_depth_profile": mode_depth.mode_key,
                "mode_depth_question_card_cap": card_cap,
                "report_style_selected": report_style,
                "report_style_fallback_used": style_fallback_used,
                "task_intent_selected": task_intent,
                "task_intent_fallback_used": intent_fallback_used,
                "complexity_tier_selected": complexity_tier,
                "complexity_tier_fallback_used": complexity_fallback_used,
                "mode_depth_adaptive_applied": adaptive_applied,
            },
        )
        return ctx

    def _build_theme_messages(
        self,
        ctx: ResearchStepContext,
        *,
        now_utc: datetime,
        card_cap: int,
        report_style: ReportStyle,
    ) -> list[dict[str, str]]:
        budget = ctx.runtime.budget
        mode_depth = ctx.runtime.mode_depth
        return build_theme_prompt_messages(
            theme=ctx.request.themes,
            search_mode=ctx.request.search_mode,
            mode_depth_profile=mode_depth.mode_key,
            current_utc_timestamp=now_utc.isoformat(),
            current_utc_date=now_utc.date().isoformat(),
            max_rounds=budget.max_rounds,
            max_search_calls=budget.max_search_calls,
            max_queries_per_round=budget.max_queries_per_round,
            card_cap=card_cap,
            hinted_style=report_style,
        )

    def _normalize_question_cards(
        self,
        raw: list[ThemeQuestionCardPayload],
        *,
        core_question: str,
        card_cap: int,
        seed_limit: int,
        fallback_branches: list[str],
    ) -> list[ResearchQuestionCard]:
        out: list[ResearchQuestionCard] = []
        seen: set[str] = set()
        for item in raw:
            question = item.question.strip()
            if not question:
                continue
            key = question.casefold()
            if key in seen:
                continue
            seen.add(key)
            priority = max(1, min(5, item.priority))
            seed_queries = normalize_strings(item.seed_queries, limit=8)
            if not seed_queries:
                seed_queries = [question]
            out.append(
                ResearchQuestionCard(
                    question_id=f"q{len(out) + 1}",
                    question=question,
                    priority=priority,
                    seed_queries=merge_strings(
                        seed_queries,
                        limit=min(seed_limit, 8),
                    ),
                    evidence_focus=normalize_strings(item.evidence_focus, limit=8),
                    expected_gain=item.expected_gain.strip()
                    or "Increase evidence coverage for this question.",
                )
            )
            if len(out) >= card_cap:
                break
        if out:
            return out
        return self._build_cards_from_fallback(
            core_question=core_question,
            fallback_branches=fallback_branches,
            card_cap=card_cap,
            seed_limit=seed_limit,
        )

    def _build_cards_from_fallback(
        self,
        *,
        core_question: str,
        fallback_branches: list[str],
        card_cap: int,
        seed_limit: int,
    ) -> list[ResearchQuestionCard]:
        out: list[ResearchQuestionCard] = []
        seen: set[str] = set()
        for branch in fallback_branches:
            question = branch.strip()
            if not question:
                continue
            if ":" in question:
                _, tail = question.split(":", 1)
                question = tail.strip() or question
            key = question.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ResearchQuestionCard(
                    question_id=f"q{len(out) + 1}",
                    question=question,
                    priority=3,
                    seed_queries=merge_strings(
                        [question], [core_question], limit=seed_limit
                    ),
                    evidence_focus=[],
                    expected_gain="Increase topic coverage.",
                )
            )
            if len(out) >= card_cap:
                return out
        if out:
            return out
        return [
            ResearchQuestionCard(
                question_id="q1",
                question=core_question,
                priority=5,
                seed_queries=[core_question],
                evidence_focus=[],
                expected_gain="Fallback single-track research for the core question.",
            )
        ]

    def _normalize_task_intent(
        self,
        *,
        raw: TaskIntent | str | None,
        theme: str,
        report_style: ReportStyle,
    ) -> TaskIntent:
        token = str(raw or "").strip().casefold().replace("-", "_")
        mapping: dict[str, TaskIntent] = {
            "how_to": "how_to",
            "howto": "how_to",
            "comparison": "comparison",
            "compare": "comparison",
            "explainer": "explainer",
            "diagnosis": "diagnosis",
            "other": "other",
        }
        if token in mapping:
            return mapping[token]
        return self._fallback_task_intent(theme=theme, report_style=report_style)

    def _fallback_task_intent(
        self,
        *,
        theme: str,
        report_style: ReportStyle,
    ) -> TaskIntent:
        token = theme.strip().casefold()
        if not token:
            return "other"
        diagnosis_hints = (
            "error",
            "issue",
            "bug",
            "troubleshoot",
            "root cause",
            "why failed",
            "failure",
            "diagnose",
        )
        comparison_hints = (
            " vs ",
            " versus ",
            "compare",
            "comparison",
            "which",
            "better",
            "best",
            "choose",
            "selection",
            "trade-off",
            "tradeoff",
        )
        how_to_hints = (
            "how to",
            "how do i",
            "step",
            "guide",
            "tutorial",
            "setup",
            "install",
            "use ",
            "workflow",
            "runbook",
            "playbook",
        )
        explainer_hints = (
            "what is",
            "explain",
            "principle",
            "mechanism",
            "overview",
            "introduction",
        )
        if any(item in token for item in diagnosis_hints):
            return "diagnosis"
        if any(item in token for item in comparison_hints):
            return "comparison"
        if any(item in token for item in how_to_hints):
            return "how_to"
        if any(item in token for item in explainer_hints):
            return "explainer"
        if report_style == "decision":
            return "comparison"
        if report_style == "execution":
            return "how_to"
        if report_style == "explainer":
            return "explainer"
        return "other"

    def _normalize_task_complexity(
        self,
        *,
        raw: TaskComplexity | str | None,
        theme: str,
        task_intent: TaskIntent,
    ) -> TaskComplexity:
        token = str(raw or "").strip().casefold()
        if token in {"low", "medium", "high"}:
            return token  # type: ignore[return-value]
        return self._fallback_task_complexity(theme=theme, task_intent=task_intent)

    def _fallback_task_complexity(
        self,
        *,
        theme: str,
        task_intent: TaskIntent,
    ) -> TaskComplexity:
        token = theme.strip().casefold()
        if not token:
            return "medium"
        high_hints = (
            "enterprise",
            "compliance",
            "governance",
            "architecture",
            "multi-tenant",
            "multi tenant",
            "regulated",
            "security",
            "threat model",
            "migration",
            "multi-region",
            "sre",
        )
        medium_hints = (
            "compare",
            "vs",
            "versus",
            "trade-off",
            "tradeoff",
            "benchmark",
            "production",
            "deploy",
            "integration",
            "cost",
            "performance",
            "scale",
            "reliability",
            "team",
            "workflow",
            "best practice",
        )
        if any(item in token for item in high_hints):
            return "high"
        if token.count(" vs ") >= 2 or token.count(" versus ") >= 2:
            return "high"
        if any(item in token for item in medium_hints):
            return "medium"
        if task_intent == "how_to":
            return "low"
        if task_intent in {"comparison", "diagnosis"}:
            return "medium"
        return "medium"

    def _apply_mode_adaptive_depth(
        self,
        *,
        ctx: ResearchStepContext,
        complexity_tier: TaskComplexity,
    ) -> bool:
        mode_depth = ctx.runtime.mode_depth
        mode_key = mode_depth.mode_key
        if mode_key not in {"research", "research-pro"}:
            return False
        if mode_key == "research":
            if complexity_tier == "high":
                return False
            if complexity_tier == "low":
                self._apply_mode_depth_overrides(
                    mode_depth=mode_depth,
                    max_question_cards_effective=2,
                    min_rounds_per_track=1,
                    gap_closure_passes=0,
                    density_gate_passes=0,
                    overview_source_topk=14,
                    content_source_topk=9,
                    content_source_chars=8_500,
                    explore_target_pages_per_round=3,
                    explore_links_per_page=7,
                )
                return True
            self._apply_mode_depth_overrides(
                mode_depth=mode_depth,
                max_question_cards_effective=3,
                min_rounds_per_track=2,
                gap_closure_passes=0,
                density_gate_passes=1,
                overview_source_topk=16,
                content_source_topk=10,
                content_source_chars=9_000,
                explore_target_pages_per_round=4,
                explore_links_per_page=9,
            )
            return True
        if complexity_tier == "high":
            return False
        if complexity_tier == "low":
            self._apply_mode_depth_overrides(
                mode_depth=mode_depth,
                max_question_cards_effective=5,
                min_rounds_per_track=3,
                gap_closure_passes=1,
                density_gate_passes=1,
                overview_source_topk=24,
                content_source_topk=17,
                content_source_chars=13_500,
                explore_target_pages_per_round=5,
                explore_links_per_page=14,
            )
            return True
        return False

    def _apply_mode_depth_overrides(
        self,
        *,
        mode_depth: ResearchModeDepthState,
        max_question_cards_effective: int,
        min_rounds_per_track: int,
        gap_closure_passes: int,
        density_gate_passes: int,
        overview_source_topk: int,
        content_source_topk: int,
        content_source_chars: int,
        explore_target_pages_per_round: int,
        explore_links_per_page: int,
    ) -> None:
        mode_depth.max_question_cards_effective = max_question_cards_effective
        mode_depth.min_rounds_per_track = min_rounds_per_track
        mode_depth.gap_closure_passes = gap_closure_passes
        mode_depth.density_gate_passes = density_gate_passes
        mode_depth.overview_source_topk = overview_source_topk
        mode_depth.content_source_topk = content_source_topk
        mode_depth.content_source_chars = content_source_chars
        mode_depth.explore_target_pages_per_round = explore_target_pages_per_round
        mode_depth.explore_links_per_page = explore_links_per_page


__all__ = ["ResearchThemeStep"]
