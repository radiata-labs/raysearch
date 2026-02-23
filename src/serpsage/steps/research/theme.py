from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    add_error,
    chat_json,
    merge_strings,
    normalize_strings,
    resolve_research_model,
)

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class ResearchThemeStep(StepBase[ResearchStepContext]):
    span_name = "step.research_theme"

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "core_question",
                "subthemes",
                "evidence_targets",
                "risk_conflicts",
                "initial_strategy",
                "seed_queries",
            ],
            "properties": {
                "core_question": {"type": "string"},
                "subthemes": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "evidence_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 12,
                },
                "risk_conflicts": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "initial_strategy": {
                    "type": "string",
                    "enum": ["coverage-first", "balanced", "depth-first"],
                },
                "seed_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max(6, int(ctx.runtime.budget.max_queries_per_round) * 3),
                },
            },
        }
        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        messages = _build_theme_messages(ctx)
        payload: dict[str, Any] = {}
        try:
            payload = await chat_json(
                ctx=ctx,
                llm=self._llm,
                model=model,
                messages=messages,
                schema=schema,
                retries=int(self.settings.research.llm_self_heal_retries),
            )
        except Exception as exc:  # noqa: BLE001
            add_error(
                ctx,
                code="research_theme_plan_failed",
                message=str(exc),
                details={},
            )

        subthemes = normalize_strings(payload.get("subthemes"), limit=12)
        seed_queries = normalize_strings(
            payload.get("seed_queries"),
            limit=max(6, int(ctx.runtime.budget.max_queries_per_round) * 3),
        )
        if not seed_queries:
            seed_queries = [ctx.request.themes]
        ctx.plan.theme_plan = {
            "core_question": str(payload.get("core_question") or ctx.request.themes),
            "subthemes": subthemes,
            "evidence_targets": normalize_strings(payload.get("evidence_targets"), limit=12),
            "risk_conflicts": normalize_strings(payload.get("risk_conflicts"), limit=10),
            "initial_strategy": str(payload.get("initial_strategy") or "balanced"),
        }
        ctx.plan.next_queries = merge_strings(
            seed_queries,
            [ctx.request.themes],
            limit=int(ctx.runtime.budget.max_queries_per_round),
        )
        ctx.corpus.coverage_state.total_subthemes = int(len(subthemes))
        ctx.notes.append(
            f"Theme plan built with {len(subthemes)} subthemes and {len(ctx.plan.next_queries)} seed queries."
        )

        span.set_attr("subthemes", int(len(subthemes)))
        span.set_attr("seed_queries", int(len(ctx.plan.next_queries)))
        return ctx


def _build_theme_messages(ctx: ResearchStepContext) -> list[dict[str, str]]:
    budget = ctx.runtime.budget
    return [
        {
            "role": "system",
            "content": (
                "Role: Senior Research Architect.\n"
                "Mission: Convert the user theme into an executable research blueprint.\n"
                "Hard Constraints:\n"
                "1) Every subtheme must be externally verifiable by search evidence.\n"
                "2) Seed queries must be concrete and non-overlapping.\n"
                "3) Prioritize factual discoverability over rhetorical phrasing.\n"
                "4) Do not output markdown or commentary. JSON only.\n"
                "Allowed Evidence:\n"
                "- User theme and search mode only.\n"
                "Failure Policy:\n"
                "- If the theme is broad, decompose into measurable subthemes.\n"
                "- If ambiguity exists, represent alternatives in risk_conflicts.\n"
                "Quality Checklist:\n"
                "- High coverage, low query redundancy, explicit evidence targets."
            ),
        },
        {
            "role": "user",
            "content": (
                f"THEME:\n{ctx.request.themes}\n\n"
                f"SEARCH_MODE:\n{ctx.request.search_mode}\n\n"
                "BUDGET_HINTS:\n"
                f"- max_rounds={budget.max_rounds}\n"
                f"- max_search_calls={budget.max_search_calls}\n"
                f"- max_queries_per_round={budget.max_queries_per_round}\n\n"
                "Required Output Schema Notes:\n"
                "- core_question: one sentence\n"
                "- subthemes: prioritized list\n"
                "- evidence_targets: source types to seek\n"
                "- risk_conflicts: likely contradiction axes\n"
                "- initial_strategy: coverage-first|balanced|depth-first\n"
                "- seed_queries: practical search queries"
            ),
        },
    ]


# Backward-compatible alias.
ResearchThemePlanStep = ResearchThemeStep


__all__ = ["ResearchThemeStep", "ResearchThemePlanStep"]

