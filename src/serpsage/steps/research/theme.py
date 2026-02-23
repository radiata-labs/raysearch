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
from serpsage.utils import clean_whitespace

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
                "detected_input_language",
                "core_question",
                "subthemes",
                "evidence_targets",
                "risk_conflicts",
                "initial_strategy",
                "seed_queries",
            ],
            "properties": {
                "detected_input_language": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 48,
                },
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

        input_language = clean_whitespace(str(payload.get("detected_input_language") or ""))
        if not input_language:
            input_language = "same as user input language"
        ctx.plan.input_language = input_language
        ctx.plan.output_language = input_language

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
            "input_language": input_language,
            "output_language": input_language,
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
        ctx.notes.append(
            f"Output language fixed to {ctx.plan.output_language}."
        )

        span.set_attr("subthemes", int(len(subthemes)))
        span.set_attr("seed_queries", int(len(ctx.plan.next_queries)))
        span.set_attr("input_language", str(ctx.plan.input_language))
        span.set_attr("output_language", str(ctx.plan.output_language))
        return ctx


def _build_theme_messages(ctx: ResearchStepContext) -> list[dict[str, str]]:
    budget = ctx.runtime.budget
    return [
        {
            "role": "system",
            "content": (
                "Role: Senior Research Architect and Curriculum Designer.\n"
                "Mission: Convert the user theme into a rigorous, executable research curriculum.\n"
                "Instruction Priority:\n"
                "P1) Schema correctness.\n"
                "P2) Evidence-seeking quality and decomposition precision.\n"
                "P3) Output language consistency.\n"
                "Hard Constraints:\n"
                "0) Detect the language of THEME and set detected_input_language accordingly.\n"
                "1) Every subtheme must be externally verifiable by search evidence.\n"
                "2) Seed queries must be concrete, non-overlapping, and high-yield.\n"
                "3) Prioritize factual discoverability over rhetorical phrasing.\n"
                "4) Avoid generic terms unless paired with disambiguating qualifiers.\n"
                "5) Free-text fields must be written in the detected_input_language.\n"
                "6) Do not output markdown or commentary. JSON only.\n"
                "Allowed Evidence:\n"
                "- User theme, search mode, and budget hints only.\n"
                "Failure Policy:\n"
                "- If the theme is broad, decompose into measurable subthemes.\n"
                "- If ambiguity exists, represent alternatives in risk_conflicts with explicit wording.\n"
                "- If the theme is underspecified, bias toward broader coverage in seed queries.\n"
                "Quality Checklist:\n"
                "- High coverage, low query redundancy, explicit evidence targets, testable subthemes."
            ),
        },
        {
            "role": "user",
            "content": (
                f"THEME:\n{ctx.request.themes}\n\n"
                f"SEARCH_MODE:\n{ctx.request.search_mode}\n\n"
                "LANGUAGE_POLICY:\n"
                "- First infer language from THEME.\n"
                "- Then keep all free-text fields in that same language.\n"
                "- Use a precise language tag (e.g., en, zh-CN, es, fr, de, ja, ar).\n\n"
                "BUDGET_HINTS:\n"
                f"- max_rounds={budget.max_rounds}\n"
                f"- max_search_calls={budget.max_search_calls}\n"
                f"- max_queries_per_round={budget.max_queries_per_round}\n\n"
                "Required Output Schema Notes:\n"
                "- detected_input_language: language tag string\n"
                "- core_question: one sentence\n"
                "- subthemes: prioritized list\n"
                "- evidence_targets: source types to seek\n"
                "- risk_conflicts: likely contradiction axes\n"
                "- initial_strategy: coverage-first|balanced|depth-first\n"
                "- seed_queries: practical search queries\n\n"
                "Anti-patterns to avoid:\n"
                "- Rephrasing the same query with superficial word swaps.\n"
                "- Subthemes that cannot be validated by web evidence.\n"
                "- Vague risk statements without a concrete contradiction dimension."
            ),
        },
    ]


# Backward-compatible alias.
ResearchThemePlanStep = ResearchThemeStep


__all__ = ["ResearchThemeStep", "ResearchThemePlanStep"]
