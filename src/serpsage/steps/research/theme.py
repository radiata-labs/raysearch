from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.pipeline import ResearchQuestionCard, ResearchStepContext
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
from serpsage.steps.research.prompt import (
    build_theme_messages as build_theme_prompt_messages,
)
from serpsage.steps.research.prompt import (
    infer_report_style_from_theme,
    resolve_report_style,
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


class ResearchThemeStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        mode_depth = ctx.runtime.mode_depth
        prompt_card_cap = max(1, int(mode_depth.max_question_cards_effective))
        seed_limit = max(6, int(ctx.runtime.budget.max_queries_per_round) * 3)
        style_cfg = self.settings.research.report_style
        hinted_style = infer_report_style_from_theme(
            ctx.request.themes,
            default=self._normalize_style_fallback(style_cfg.fallback_style),
        )
        fallback_task_intent = self._normalize_task_intent(
            raw=None,
            theme=ctx.request.themes,
            report_style=hinted_style,
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
            detected_input_language="same as user input language",
            core_question=ctx.request.themes,
            report_style=hinted_style,
            task_intent=fallback_task_intent,
            complexity_tier=fallback_complexity,
            subthemes=[],
            required_entities=[],
            question_cards=[],
        )
        try:
            chat_result = await self._llm.chat(
                model=model,
                messages=self._build_theme_messages(
                    ctx,
                    now_utc=now_utc,
                    card_cap=prompt_card_cap,
                ),
                response_format=ThemeOutputPayload,
                format_override=self._build_theme_schema(card_cap=prompt_card_cap),
                retries=int(self.settings.research.llm_self_heal_retries),
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
                    "model": str(model),
                    "message": str(exc),
                },
            )
        input_language = clean_whitespace(str(payload.detected_input_language or ""))
        if not input_language:
            input_language = "same as user input language"
        core_question = clean_whitespace(
            str(payload.core_question or ctx.request.themes)
        )
        if not core_question:
            core_question = ctx.request.themes
        report_style = resolve_report_style(
            raw_style=payload.report_style,
            theme=core_question or ctx.request.themes,
            enabled=bool(style_cfg.enabled),
            fallback_style=self._normalize_style_fallback(style_cfg.fallback_style),
            strict_style_lock=bool(style_cfg.strict_style_lock),
        )
        raw_style_token = clean_whitespace(str(payload.report_style or "")).casefold()
        style_fallback_used = bool(raw_style_token != str(report_style))
        task_intent = self._normalize_task_intent(
            raw=payload.task_intent,
            theme=core_question or ctx.request.themes,
            report_style=report_style,
        )
        raw_task_intent_token = clean_whitespace(
            str(payload.task_intent or "")
        ).casefold()
        intent_fallback_used = bool(raw_task_intent_token != str(task_intent))
        complexity_tier = self._normalize_task_complexity(
            raw=payload.complexity_tier,
            theme=core_question or ctx.request.themes,
            task_intent=task_intent,
        )
        raw_complexity_token = clean_whitespace(
            str(payload.complexity_tier or "")
        ).casefold()
        complexity_fallback_used = bool(raw_complexity_token != str(complexity_tier))
        adaptive_applied = self._apply_pro_adaptive_mode_depth(
            ctx=ctx,
            complexity_tier=complexity_tier,
        )
        mode_depth = ctx.runtime.mode_depth
        card_cap = max(1, int(mode_depth.max_question_cards_effective))
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
        ctx.plan.input_language = input_language
        ctx.plan.output_language = input_language
        ctx.plan.core_question = core_question
        ctx.parallel.question_cards = [item.model_copy(deep=True) for item in cards]
        ctx.plan.theme_plan = ResearchThemePlan(
            core_question=core_question,
            report_style=report_style,
            task_intent=task_intent,
            complexity_tier=complexity_tier,
            subthemes=subthemes,
            required_entities=required_entities,
            input_language=input_language,
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
        seed_groups = [list(item.seed_queries) for item in cards]
        next_query_limit = max(8, int(ctx.runtime.budget.max_queries_per_round) * 3)
        ctx.plan.next_queries = merge_strings(
            *seed_groups,
            [core_question],
            limit=next_query_limit,
        )
        ctx.corpus.coverage_state.total_subthemes = int(len(subthemes))
        ctx.notes.append(
            f"Theme plan built with {len(cards)} question cards and {len(subthemes)} subthemes."
        )
        ctx.notes.append(f"Report style fixed to `{report_style}`.")
        ctx.notes.append(
            f"Task intent fixed to `{task_intent}` with complexity tier `{complexity_tier}`."
        )
        if adaptive_applied:
            ctx.notes.append(
                "Adaptive research-pro depth applied based on theme complexity."
            )
        if required_entities:
            ctx.notes.append(f"Required entities: {', '.join(required_entities[:8])}.")
        ctx.notes.append(f"Output language fixed to {ctx.plan.output_language}.")
        await self.emit_tracking_event(
            event_name="research.style.selected",
            request_id=ctx.request_id,
            stage="theme_plan",
            attrs={
                "report_style_selected": str(report_style),
                "report_style_fallback_used": bool(style_fallback_used),
                "task_intent_selected": str(task_intent),
                "task_intent_fallback_used": bool(intent_fallback_used),
                "complexity_tier_selected": str(complexity_tier),
                "complexity_tier_fallback_used": bool(complexity_fallback_used),
            },
        )
        if clean_whitespace(str(mode_depth.mode_key)).casefold() == "research-pro":
            await self.emit_tracking_event(
                event_name="research.mode_depth.adaptive_applied",
                request_id=ctx.request_id,
                stage="theme_plan",
                attrs={
                    "mode_depth_profile": str(mode_depth.mode_key),
                    "task_intent": str(task_intent),
                    "complexity_tier": str(complexity_tier),
                    "effective_complexity_tier": str(complexity_tier),
                    "adaptive_applied": bool(adaptive_applied),
                    "max_question_cards_effective": int(
                        mode_depth.max_question_cards_effective
                    ),
                    "min_rounds_per_track": int(mode_depth.min_rounds_per_track),
                    "gap_closure_passes": int(mode_depth.gap_closure_passes),
                    "density_gate_passes": int(mode_depth.density_gate_passes),
                    "render_section_min": int(mode_depth.render_section_min),
                    "render_section_max": int(mode_depth.render_section_max),
                    "target_length_ratio_vs_current": float(
                        mode_depth.target_length_ratio_vs_current
                    ),
                },
            )
        await self.emit_tracking_event(
            event_name="research.theme.summary",
            request_id=ctx.request_id,
            stage="theme_plan",
            attrs={
                "core_question": ctx.plan.core_question,
                "question_cards": len(cards),
                "subthemes": len(subthemes),
                "next_queries": len(ctx.plan.next_queries),
                "mode_depth_profile": str(mode_depth.mode_key),
                "mode_depth_question_card_cap": int(card_cap),
                "report_style_selected": str(report_style),
                "report_style_fallback_used": bool(style_fallback_used),
                "task_intent_selected": str(task_intent),
                "task_intent_fallback_used": bool(intent_fallback_used),
                "complexity_tier_selected": str(complexity_tier),
                "complexity_tier_fallback_used": bool(complexity_fallback_used),
                "mode_depth_adaptive_applied": bool(adaptive_applied),
            },
        )
        return ctx

    def _build_theme_messages(
        self,
        ctx: ResearchStepContext,
        *,
        now_utc: datetime,
        card_cap: int,
    ) -> list[dict[str, str]]:
        budget = ctx.runtime.budget
        mode_depth = ctx.runtime.mode_depth
        hinted_style = infer_report_style_from_theme(
            ctx.request.themes,
            default=self._normalize_style_fallback(
                self.settings.research.report_style.fallback_style
            ),
        )
        hinted_intent = self._normalize_task_intent(
            raw=None,
            theme=ctx.request.themes,
            report_style=hinted_style,
        )
        hinted_complexity = self._normalize_task_complexity(
            raw=None,
            theme=ctx.request.themes,
            task_intent=hinted_intent,
        )
        return build_theme_prompt_messages(
            theme=ctx.request.themes,
            search_mode=ctx.request.search_mode,
            mode_depth_profile=str(mode_depth.mode_key),
            current_utc_timestamp=now_utc.isoformat(),
            current_utc_date=now_utc.date().isoformat(),
            max_rounds=int(budget.max_rounds),
            max_search_calls=int(budget.max_search_calls),
            max_queries_per_round=int(budget.max_queries_per_round),
            card_cap=int(card_cap),
            hinted_style=hinted_style,
            hinted_task_intent=hinted_intent,
            hinted_complexity_tier=hinted_complexity,
        )

    def _build_theme_schema(self, *, card_cap: int) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "detected_input_language",
                "core_question",
                "report_style",
                "task_intent",
                "complexity_tier",
                "subthemes",
                "required_entities",
                "question_cards",
            ],
            "properties": {
                "detected_input_language": {"type": "string"},
                "core_question": {"type": "string"},
                "report_style": {
                    "type": "string",
                    "enum": ["decision", "explainer", "execution"],
                },
                "task_intent": {
                    "type": "string",
                    "enum": ["how_to", "comparison", "explainer", "diagnosis", "other"],
                },
                "complexity_tier": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "subthemes": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string"},
                },
                "required_entities": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {"type": "string"},
                },
                "question_cards": {
                    "type": "array",
                    "maxItems": max(1, int(card_cap)),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "question",
                            "priority",
                            "seed_queries",
                            "evidence_focus",
                            "expected_gain",
                        ],
                        "properties": {
                            "question": {"type": "string"},
                            "priority": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "seed_queries": {
                                "type": "array",
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "evidence_focus": {
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
            question = clean_whitespace(item.question)
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
                    expected_gain=clean_whitespace(item.expected_gain)
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
            question = clean_whitespace(branch)
            if not question:
                continue
            if ":" in question:
                _, tail = question.split(":", 1)
                question = clean_whitespace(tail) or question
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

    def _normalize_style_fallback(self, raw: object) -> ReportStyle:
        token = clean_whitespace(str(raw or "")).casefold()
        if token in {"decision", "explainer", "execution"}:
            return token  # type: ignore[return-value]
        return "explainer"

    def _normalize_task_intent(
        self,
        *,
        raw: object | None,
        theme: str,
        report_style: ReportStyle,
    ) -> TaskIntent:
        token = clean_whitespace(str(raw or "")).casefold().replace("-", "_")
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
        token = clean_whitespace(theme).casefold()
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
        raw: object | None,
        theme: str,
        task_intent: TaskIntent,
    ) -> TaskComplexity:
        token = clean_whitespace(str(raw or "")).casefold()
        if token in {"low", "medium", "high"}:
            return token  # type: ignore[return-value]
        return self._fallback_task_complexity(theme=theme, task_intent=task_intent)

    def _fallback_task_complexity(
        self,
        *,
        theme: str,
        task_intent: TaskIntent,
    ) -> TaskComplexity:
        token = clean_whitespace(theme).casefold()
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

    def _apply_pro_adaptive_mode_depth(
        self,
        *,
        ctx: ResearchStepContext,
        complexity_tier: TaskComplexity,
    ) -> bool:
        mode_depth = ctx.runtime.mode_depth
        if clean_whitespace(str(mode_depth.mode_key)).casefold() != "research-pro":
            return False
        if complexity_tier == "high":
            return False
        if complexity_tier == "low":
            mode_depth.max_question_cards_effective = 3
            mode_depth.min_rounds_per_track = 2
            mode_depth.gap_closure_passes = 1
            mode_depth.density_gate_passes = 1
            mode_depth.render_section_min = 6
            mode_depth.render_section_max = 7
            mode_depth.target_length_ratio_vs_current = 1.0
            return True
        mode_depth.max_question_cards_effective = 4
        mode_depth.min_rounds_per_track = 2
        mode_depth.gap_closure_passes = 1
        mode_depth.density_gate_passes = 1
        mode_depth.render_section_min = 7
        mode_depth.render_section_max = 8
        mode_depth.target_length_ratio_vs_current = 1.05
        return True


__all__ = ["ResearchThemeStep"]
