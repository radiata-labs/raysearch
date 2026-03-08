from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.steps.research import (
    ResearchModeDepthState,
    ResearchQuestionCard,
    ResearchStepContext,
    ResearchThemePlan,
    ResearchThemePlanCard,
    TaskComplexity,
    ThemeOutputPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_theme_prompt_messages
from serpsage.steps.research.schema import build_theme_schema
from serpsage.steps.research.utils import resolve_research_model

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import ResearchModeSettings

_ADAPTIVE_DEPTH_FIELDS: tuple[str, ...] = (
    "max_question_cards_effective",
    "min_rounds_per_track",
    "source_topk",
    "source_chars",
    "content_chars",
    "explore_target_pages_per_round",
    "explore_links_per_page",
)


class ResearchThemeStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
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
        card_cap = max(1, ctx.runtime.mode_depth.max_question_cards_effective)
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=build_theme_prompt_messages(
                    ctx=ctx,
                    now_utc=now_utc,
                    card_cap=card_cap,
                    report_style=ctx.plan.theme_plan.report_style,
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
            card_cap=max(1, ctx.runtime.mode_depth.max_question_cards_effective),
        )
        theme_plan = ResearchThemePlan(
            core_question=payload.core_question,
            report_style=payload.report_style,
            task_intent=payload.task_intent,
            complexity_tier=payload.complexity_tier,
            subthemes=list(payload.subthemes),
            required_entities=list(payload.required_entities),
            input_language=payload.detected_input_language,
            search_language=payload.search_language,
            output_language=payload.detected_input_language,
            question_cards=[
                ResearchThemePlanCard(
                    question_id=card.question_id,
                    question=card.question,
                    priority=card.priority,
                    seed_queries=list(card.seed_queries),
                    evidence_focus=list(card.evidence_focus),
                    expected_gain=card.expected_gain,
                )
                for card in question_cards
            ],
        )
        ctx.plan.theme_plan = theme_plan
        ctx.parallel.question_cards = [
            item.model_copy(deep=True) for item in question_cards
        ]
        ctx.corpus.coverage_state.total_subthemes = len(theme_plan.subthemes)
        ctx.notes.append(
            f"Theme plan built with {len(question_cards)} question cards and {len(theme_plan.subthemes)} subthemes."
        )
        ctx.notes.append(f"Search language fixed to {theme_plan.search_language}.")
        ctx.notes.append(f"Report style fixed to `{theme_plan.report_style}`.")
        ctx.notes.append(
            "Task intent fixed to "
            f"`{theme_plan.task_intent}` with complexity tier `{theme_plan.complexity_tier}`."
        )
        if adaptive_applied:
            ctx.notes.append(
                f"Adaptive `{ctx.runtime.mode_depth.mode_key}` depth applied based on theme complexity."
            )
        if theme_plan.required_entities:
            ctx.notes.append(
                "Required entities: "
                + ", ".join(theme_plan.required_entities[:8])
                + "."
            )
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
                "report_style_selected": theme_plan.report_style,
                "task_intent_selected": theme_plan.task_intent,
                "complexity_tier_selected": theme_plan.complexity_tier,
            },
        )
        if ctx.runtime.mode_depth.mode_key in {"research", "research-pro"}:
            await self.emit_tracking_event(
                event_name="research.mode_depth.adaptive_applied",
                request_id=ctx.request_id,
                stage="theme_plan",
                attrs={
                    "mode_depth_profile": ctx.runtime.mode_depth.mode_key,
                    "task_intent": theme_plan.task_intent,
                    "complexity_tier": theme_plan.complexity_tier,
                    "effective_complexity_tier": theme_plan.complexity_tier,
                    "adaptive_applied": adaptive_applied,
                    "max_question_cards_effective": (
                        ctx.runtime.mode_depth.max_question_cards_effective
                    ),
                    "min_rounds_per_track": ctx.runtime.mode_depth.min_rounds_per_track,
                    "source_topk": ctx.runtime.mode_depth.source_topk,
                    "explore_target_pages_per_round": (
                        ctx.runtime.mode_depth.explore_target_pages_per_round
                    ),
                },
            )
        await self.emit_tracking_event(
            event_name="research.theme.summary",
            request_id=ctx.request_id,
            stage="theme_plan",
            attrs={
                "core_question": theme_plan.core_question,
                "question_cards": len(question_cards),
                "subthemes": len(theme_plan.subthemes),
                "input_language": theme_plan.input_language,
                "search_language": theme_plan.search_language,
                "output_language": theme_plan.output_language,
                "mode_depth_profile": ctx.runtime.mode_depth.mode_key,
                "mode_depth_question_card_cap": (
                    ctx.runtime.mode_depth.max_question_cards_effective
                ),
                "report_style_selected": theme_plan.report_style,
                "task_intent_selected": theme_plan.task_intent,
                "complexity_tier_selected": theme_plan.complexity_tier,
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
        mode_depth = ctx.runtime.mode_depth
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
        mode_depth: ResearchModeDepthState,
        profile: ResearchModeSettings,
    ) -> None:
        mode_depth.max_question_cards_effective = profile.max_question_cards_effective
        mode_depth.min_rounds_per_track = profile.min_rounds_per_track
        mode_depth.source_topk = profile.source_topk
        mode_depth.source_chars = profile.source_chars
        mode_depth.content_chars = profile.content_chars
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
