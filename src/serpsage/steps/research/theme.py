from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
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
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "detected_input_language",
                "core_question",
                "multi_level_questions",
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
                "multi_level_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 24,
                },
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
        messages = self._build_theme_messages(ctx, now_utc=now_utc)
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
            ctx.errors.append(
                AppError(
                    code="research_theme_plan_failed",
                    message=str(exc),
                    details={},
                )
            )

        input_language = clean_whitespace(str(payload.get("detected_input_language") or ""))
        if not input_language:
            input_language = "same as user input language"
        ctx.plan.input_language = input_language
        ctx.plan.output_language = input_language

        multi_level_questions = normalize_strings(
            payload.get("multi_level_questions"),
            limit=24,
        )
        subthemes = normalize_strings(payload.get("subthemes"), limit=12)
        if not subthemes and multi_level_questions:
            subthemes = self._extract_subthemes_from_multi_level(
                multi_level_questions,
                limit=12,
            )
        seed_queries = normalize_strings(
            payload.get("seed_queries"),
            limit=max(6, int(ctx.runtime.budget.max_queries_per_round) * 3),
        )
        if not seed_queries:
            seed_queries = [ctx.request.themes]
        ctx.plan.theme_plan = {
            "core_question": str(payload.get("core_question") or ctx.request.themes),
            "multi_level_questions": multi_level_questions,
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
        print(
            "[research.theme]",
            json.dumps(
                {
                    "detected_input_language": ctx.plan.input_language,
                    "theme_plan": ctx.plan.theme_plan,
                    "next_queries": ctx.plan.next_queries,
                    "raw_model_payload": payload,
                },
                ensure_ascii=False,
            ),
        )
        return ctx

    def _build_theme_messages(
        self, ctx: ResearchStepContext, *, now_utc: datetime
    ) -> list[dict[str, str]]:
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
                    "1) Build a multi-level question tree, not a flat list.\n"
                    "2) multi_level_questions must include at least L1/L2 levels and optionally L3 verification checks.\n"
                    "3) Every subtheme must be externally verifiable by search evidence.\n"
                    "4) Seed queries must be concrete, non-overlapping, and high-yield.\n"
                    "5) Seed queries must jointly cover different levels of the question tree.\n"
                    "6) Prioritize factual discoverability over rhetorical phrasing.\n"
                    "7) Avoid generic terms unless paired with disambiguating qualifiers.\n"
                    "8) Free-text fields must be written in the detected_input_language.\n"
                    "9) For each major branch, include at least one verification-oriented question.\n"
                    "10) Seed queries should prioritize authoritative and content-rich evidence paths (official docs, primary data, technical reports, standards, reputable benchmarks).\n"
                    "11) For comparative themes, explicitly include head-to-head and scenario-specific questions.\n"
                    "12) Do not output markdown or commentary. JSON only.\n"
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
                    "TIME_CONTEXT:\n"
                    f"- current_utc_timestamp={now_utc.isoformat()}\n"
                    f"- current_utc_date={now_utc.date().isoformat()}\n\n"
                    "TEMPORAL_POLICY:\n"
                    "- If THEME explicitly asks for recency (latest/current/today/now/as of/this year/month/week), treat recency as mandatory.\n"
                    "- When recency is mandatory, seed_queries must include explicit time anchors or freshness qualifiers.\n"
                    "- Resolve relative time words against current_utc_date and avoid ambiguous temporal phrasing.\n"
                    "- If THEME does not request recency, do not force unnecessary date constraints.\n\n"
                    "BUDGET_HINTS:\n"
                    f"- max_rounds={budget.max_rounds}\n"
                    f"- max_search_calls={budget.max_search_calls}\n"
                    f"- max_queries_per_round={budget.max_queries_per_round}\n\n"
                    "Required Output Schema Notes:\n"
                    "- detected_input_language: language tag string\n"
                    "- core_question: one sentence\n"
                    "- multi_level_questions: hierarchical list using explicit level prefixes (e.g., L1:, L2:, L3:)\n"
                    "- subthemes: prioritized list\n"
                    "- evidence_targets: source types to seek\n"
                    "- risk_conflicts: likely contradiction axes\n"
                    "- initial_strategy: coverage-first|balanced|depth-first\n"
                    "- seed_queries: practical search queries\n\n"
                    "Question-tree depth guidance:\n"
                    "- L1: strategic decision questions.\n"
                    "- L2: evidence dimensions (performance, cost, risk, ecosystem, maintainability, recency).\n"
                    "- L3: verification checks and tie-break questions.\n\n"
                    "Anti-patterns to avoid:\n"
                    "- Rephrasing the same query with superficial word swaps.\n"
                    "- Flat decomposition without level structure.\n"
                    "- Subthemes that cannot be validated by web evidence.\n"
                    "- Vague risk statements without a concrete contradiction dimension."
                ),
            },
        ]

    def _extract_subthemes_from_multi_level(
        self,
        values: list[str],
        *,
        limit: int,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            item = clean_whitespace(raw)
            if not item:
                continue
            normalized = item
            if ":" in normalized:
                _, tail = normalized.split(":", 1)
                normalized = clean_whitespace(tail)
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
            if len(out) >= max(1, int(limit)):
                break
        return out


# Backward-compatible alias.
ResearchThemePlanStep = ResearchThemeStep


__all__ = ["ResearchThemeStep", "ResearchThemePlanStep"]
