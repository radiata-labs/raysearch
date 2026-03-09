from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.core.runtime import Runtime
from serpsage.dependencies import Inject
from serpsage.models.steps.research import (
    ResearchLimits,
    ResearchQuestionCard,
    ResearchStepContext,
    ResearchTask,
    TaskComplexity,
    ThemeOutputPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_theme_prompt_messages
from serpsage.steps.research.schema import build_theme_schema
from serpsage.steps.research.utils import resolve_research_model

if TYPE_CHECKING:
    from serpsage.settings.models import ResearchModeSettings

_ADAPTIVE_DEPTH_FIELDS: tuple[str, ...] = (
    "max_question_cards_effective",
    "min_rounds_per_track",
    "round_search_budget",
    "round_fetch_budget",
    "review_source_window",
    "report_source_batch_size",
    "report_source_batch_chars",
    "fetch_page_max_chars",
    "explore_target_pages_per_round",
    "explore_links_per_page",
)


class ResearchThemeStep(StepBase[ResearchStepContext]):
    def __init__(
        self, *, rt: Runtime = Inject(), llm: LLMClientBase = Inject()
    ) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        card_cap = max(1, ctx.run.limits.max_question_cards_effective)
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=build_theme_prompt_messages(
                    ctx=ctx,
                    now_utc=now_utc,
                    card_cap=card_cap,
                    report_style=ctx.task.style,
                ),
                response_format=ThemeOutputPayload,
                format_override=build_theme_schema(card_cap=card_cap),
                retries=self.settings.research.llm_self_heal_retries,
            )
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
            raise
        payload = chat_result.data
        adaptive_applied = self._apply_mode_adaptive_depth(
            ctx=ctx,
            complexity_tier=payload.complexity_tier,
        )
        question_cards = self._build_question_cards(
            payload=payload,
            card_cap=max(1, ctx.run.limits.max_question_cards_effective),
        )
        ctx.task = ResearchTask(
            question=payload.core_question,
            style=payload.report_style,
            intent=payload.task_intent,
            complexity=payload.complexity_tier,
            subthemes=list(payload.subthemes),
            entities=list(payload.required_entities),
            input_language=payload.detected_input_language,
            output_language=payload.detected_input_language,
            cards=[item.model_copy(deep=True) for item in question_cards],
        )
        theme_plan = ctx.task
        ctx.run.notes.append(
            f"Theme plan built with {len(question_cards)} question cards and {len(theme_plan.subthemes)} subthemes."
        )
        ctx.run.notes.append(f"Report style fixed to `{theme_plan.style}`.")
        ctx.run.notes.append(
            "Task intent fixed to "
            f"`{theme_plan.intent}` with complexity tier `{theme_plan.complexity}`."
        )
        if adaptive_applied:
            ctx.run.notes.append(
                f"Adaptive `{ctx.run.limits.mode_key}` depth applied based on theme complexity."
            )
        if theme_plan.entities:
            ctx.run.notes.append(
                "Required entities: " + ", ".join(theme_plan.entities[:8]) + "."
            )
        ctx.run.notes.append(f"Output language fixed to {theme_plan.output_language}.")
        await self.emit_tracking_event(
            event_name="research.language.selected",
            request_id=ctx.request_id,
            stage="theme_plan",
            attrs={
                "input_language": theme_plan.input_language,
                "output_language": theme_plan.output_language,
            },
        )
        await self.emit_tracking_event(
            event_name="research.style.selected",
            request_id=ctx.request_id,
            stage="theme_plan",
            attrs={
                "report_style_selected": theme_plan.style,
                "task_intent_selected": theme_plan.intent,
                "complexity_tier_selected": theme_plan.complexity,
            },
        )
        if ctx.run.limits.mode_key in {"research", "research-pro"}:
            await self.emit_tracking_event(
                event_name="research.mode_depth.adaptive_applied",
                request_id=ctx.request_id,
                stage="theme_plan",
                attrs={
                    "mode_depth_profile": ctx.run.limits.mode_key,
                    "task_intent": theme_plan.intent,
                    "complexity_tier": theme_plan.complexity,
                    "effective_complexity_tier": theme_plan.complexity,
                    "adaptive_applied": adaptive_applied,
                    "max_question_cards_effective": ctx.run.limits.max_question_cards_effective,
                    "min_rounds_per_track": ctx.run.limits.min_rounds_per_track,
                    "round_search_budget": ctx.run.limits.round_search_budget,
                    "round_fetch_budget": ctx.run.limits.round_fetch_budget,
                    "review_source_window": ctx.run.limits.review_source_window,
                    "report_source_batch_size": ctx.run.limits.report_source_batch_size,
                    "explore_target_pages_per_round": ctx.run.limits.explore_target_pages_per_round,
                },
            )
        await self.emit_tracking_event(
            event_name="research.theme.summary",
            request_id=ctx.request_id,
            stage="theme_plan",
            attrs={
                "core_question": theme_plan.question,
                "question_cards": len(question_cards),
                "subthemes": len(theme_plan.subthemes),
                "input_language": theme_plan.input_language,
                "output_language": theme_plan.output_language,
                "mode_depth_profile": ctx.run.limits.mode_key,
                "mode_depth_question_card_cap": ctx.run.limits.max_question_cards_effective,
                "report_style_selected": theme_plan.style,
                "task_intent_selected": theme_plan.intent,
                "complexity_tier_selected": theme_plan.complexity,
                "mode_depth_adaptive_applied": adaptive_applied,
            },
        )
        return ctx

    def _build_question_cards(
        self,
        *,
        payload: ThemeOutputPayload,
        card_cap: int,
    ) -> list[ResearchQuestionCard]:
        return [
            ResearchQuestionCard(
                question_id=f"q{index}",
                question=item.question,
                priority=item.priority,
                seed_queries=list(item.seed_queries),
                evidence_focus=list(item.evidence_focus),
                expected_gain=item.expected_gain,
            )
            for index, item in enumerate(payload.question_cards[:card_cap], start=1)
        ]

    def _apply_mode_adaptive_depth(
        self,
        *,
        ctx: ResearchStepContext,
        complexity_tier: TaskComplexity,
    ) -> bool:
        mode_depth = ctx.run.limits
        mode_key = mode_depth.mode_key
        if mode_key not in {"research", "research-pro"}:
            return False
        if complexity_tier == "high":
            return False
        current_profile = self._resolve_mode_profile(mode_key)
        lower_profile = self._resolve_mode_profile(
            "research-fast" if mode_key == "research" else "research"
        )
        if mode_key == "research":
            low_profile = self._build_midpoint_profile(lower_profile, current_profile)
            target_profile = (
                low_profile
                if complexity_tier == "low"
                else self._build_midpoint_profile(low_profile, current_profile)
            )
        else:
            target_profile = (
                lower_profile
                if complexity_tier == "low"
                else self._build_midpoint_profile(lower_profile, current_profile)
            )
        self._apply_mode_depth_overrides(
            mode_depth=mode_depth,
            profile=target_profile,
        )
        return True

    def _apply_mode_depth_overrides(
        self,
        *,
        mode_depth: ResearchLimits,
        profile: ResearchModeSettings,
    ) -> None:
        mode_depth.max_question_cards_effective = profile.max_question_cards_effective
        mode_depth.min_rounds_per_track = profile.min_rounds_per_track
        mode_depth.round_search_budget = profile.round_search_budget
        mode_depth.round_fetch_budget = profile.round_fetch_budget
        mode_depth.review_source_window = profile.review_source_window
        mode_depth.report_source_batch_size = profile.report_source_batch_size
        mode_depth.report_source_batch_chars = profile.report_source_batch_chars
        mode_depth.fetch_page_max_chars = profile.fetch_page_max_chars
        mode_depth.explore_target_pages_per_round = (
            profile.explore_target_pages_per_round
        )
        mode_depth.explore_links_per_page = profile.explore_links_per_page

    def _resolve_mode_profile(self, mode_key: str) -> ResearchModeSettings:
        profiles: dict[str, ResearchModeSettings] = {
            "research-fast": self.settings.research.research_fast,
            "research": self.settings.research.research,
            "research-pro": self.settings.research.research_pro,
        }
        return profiles[mode_key]

    def _build_midpoint_profile(
        self,
        left: ResearchModeSettings,
        right: ResearchModeSettings,
    ) -> ResearchModeSettings:
        updates = {
            field_name: self._midpoint_int(
                int(getattr(left, field_name)),
                int(getattr(right, field_name)),
            )
            for field_name in _ADAPTIVE_DEPTH_FIELDS
        }
        return right.model_copy(update=updates)

    def _midpoint_int(self, left: int, right: int) -> int:
        return (int(left) + int(right)) // 2


__all__ = ["ResearchThemeStep"]
