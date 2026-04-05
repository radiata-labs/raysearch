from __future__ import annotations

from datetime import UTC, datetime
from typing_extensions import override

from raysearch.components.llm.base import LLMClientBase
from raysearch.components.provider.base import SearchProviderBase
from raysearch.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from raysearch.dependencies import Depends
from raysearch.models.steps.research import (
    ResearchStepContext,
    ResearchTask,
)
from raysearch.models.steps.research.payloads import (
    ResearchThemePlanCard,
    ThemeOutputPayload,
)
from raysearch.steps.base import StepBase
from raysearch.steps.research.prompt import build_theme_prompt_messages
from raysearch.steps.research.schema import build_theme_schema
from raysearch.steps.research.utils import resolve_research_model


class ResearchThemeStep(StepBase[ResearchStepContext]):
    llm: LLMClientBase = Depends()
    provider: SearchProviderBase = Depends()

    @override
    async def should_run(self, ctx: ResearchStepContext) -> bool:
        """Theme planning always runs (first step in research pipeline)."""
        _ = ctx
        return True

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        model = resolve_research_model(
            settings=self.settings,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        card_cap = max(1, ctx.run.limits.max_question_cards_effective)
        routes = resolve_engine_selection_routes(
            settings=self.settings,
            subsystem="research",
            provider=self.provider,
        )
        engine_selection_context = build_engine_selection_context(routes=routes)
        try:
            chat_result = await self.llm.create(
                model=model,
                messages=build_theme_prompt_messages(
                    ctx=ctx,
                    now_utc=now_utc,
                    card_cap=card_cap,
                    report_style=ctx.task.style,
                    engine_selection_context=engine_selection_context,
                ),
                response_format=ThemeOutputPayload,
                format_override=build_theme_schema(
                    card_cap=card_cap,
                    select_engines=bool(routes),
                ),
                retries=self.settings.research.llm_self_heal_retries,
            )
            await self.meter.record(
                name="llm.tokens",
                request_id=ctx.request_id,
                model=str(model),
                unit="token",
                tokens={
                    "prompt_tokens": int(chat_result.usage.prompt_tokens),
                    "completion_tokens": int(chat_result.usage.completion_tokens),
                    "total_tokens": int(chat_result.usage.total_tokens),
                },
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="research.theme.failed",
                request_id=ctx.request_id,
                step="research.theme",
                error_code="research_theme_plan_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "model": model,
                },
            )
            raise
        payload = chat_result.data
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
        if theme_plan.entities:
            ctx.run.notes.append(
                "Required entities: " + ", ".join(theme_plan.entities[:8]) + "."
            )
        ctx.run.notes.append(
            f"Mode depth kept from requested search mode `{ctx.run.limits.mode_key}`."
        )
        ctx.run.notes.append(f"Output language fixed to {theme_plan.output_language}.")
        await self.tracker.info(
            name="research.theme.summary",
            request_id=ctx.request_id,
            step="research.theme",
            data={
                "success": True,
                "question_cards": len(question_cards),
                "report_style_selected": theme_plan.style,
                "output_language": theme_plan.output_language,
            },
        )
        await self.tracker.debug(
            name="research.theme.summary.detail",
            request_id=ctx.request_id,
            step="research.theme",
            data={
                "success": True,
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
                "required_entities_count": len(theme_plan.entities),
            },
        )
        return ctx

    def _build_question_cards(
        self,
        *,
        payload: ThemeOutputPayload,
        card_cap: int,
    ) -> list[ResearchThemePlanCard]:
        return [
            ResearchThemePlanCard(
                question_id=f"q{index}",
                question=item.question,
                priority=item.priority,
                # QuerySourceSpec fields are primitives or list[str]; shallow copy suffices.
                seed_queries=[query.model_copy() for query in list(item.seed_queries)],
                evidence_focus=list(item.evidence_focus),
                expected_gain=item.expected_gain,
            )
            for index, item in enumerate(payload.question_cards[:card_cap], start=1)
        ]


__all__ = ["ResearchThemeStep"]
