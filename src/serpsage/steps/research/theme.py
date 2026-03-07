from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from typing_extensions import override

from serpsage.steps.base import StepBase
from serpsage.steps.models import (
    ResearchModeDepthState,
    ResearchQuestionCard,
    ResearchStepContext,
)
from serpsage.steps.research.language import (
    detect_input_language,
    normalize_language_code,
    route_search_language,
)
from serpsage.steps.research.payloads import (
    ReportStyle,
    ResearchThemePlan,
    ResearchThemePlanCard,
    TaskComplexity,
    TaskIntent,
    ThemeOutputPayload,
    ThemeQuestionCardPayload,
)
from serpsage.steps.research.prompt import build_theme_prompt_messages
from serpsage.steps.research.schema import build_theme_schema
from serpsage.steps.research.utils import (
    merge_strings,
    normalize_strings,
    resolve_research_model,
)

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import ResearchModeSettings

_STYLE_VALUES: set[str] = {"decision", "explainer", "execution"}
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
        mode_depth = ctx.runtime.mode_depth
        prompt_card_cap = max(1, mode_depth.max_question_cards_effective)
        seed_limit = max(6, ctx.runtime.budget.max_queries_per_round * 3)
        fallback_report_style = self._resolve_report_style(theme=ctx.request.themes)
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
                messages=build_theme_prompt_messages(
                    ctx=ctx,
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
        search_language = normalize_language_code(
            payload.search_language,
            default="other",
        )
        if search_language == "other":
            search_language = route_search_language(
                theme=core_question,
                input_language=input_language,
            )
        report_style = self._resolve_report_style(
            theme=core_question,
            raw_style=payload.report_style,
        )
        task_intent = self._normalize_task_intent(
            raw=payload.task_intent,
            theme=core_question,
            report_style=report_style,
        )
        complexity_tier = self._normalize_task_complexity(
            raw=payload.complexity_tier,
            theme=core_question,
            task_intent=task_intent,
        )
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
                "task_intent_selected": task_intent,
                "complexity_tier_selected": complexity_tier,
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
                    "source_topk": mode_depth.source_topk,
                    "explore_target_pages_per_round": (
                        mode_depth.explore_target_pages_per_round
                    ),
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
                "input_language": theme_plan.input_language,
                "search_language": theme_plan.search_language,
                "output_language": theme_plan.output_language,
                "mode_depth_profile": mode_depth.mode_key,
                "mode_depth_question_card_cap": card_cap,
                "report_style_selected": report_style,
                "task_intent_selected": task_intent,
                "complexity_tier_selected": complexity_tier,
                "mode_depth_adaptive_applied": adaptive_applied,
            },
        )
        return ctx

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

    def _resolve_report_style(
        self,
        *,
        theme: str,
        raw_style: str | None = None,
    ) -> ReportStyle:
        candidate = str(raw_style).strip().casefold()
        if candidate in _STYLE_VALUES:
            return cast("ReportStyle", candidate)
        token = theme.strip().casefold()
        if not token:
            return "explainer"
        decision_hints = (
            "vs",
            "versus",
            "compare",
            "comparison",
            "choice",
            "choose",
            "select",
            "selection",
            "best",
            "better",
            "recommend",
            "which one",
            "trade-off",
            "tradeoff",
        )
        execution_hints = (
            "how do i",
            "how to",
            "step by step",
            "step",
            "guide",
            "workflow",
            "implementation",
            "implement",
            "execute",
            "playbook",
            "runbook",
            "operational",
            "operation",
            "deploy",
            "deployment",
            "migration",
            "rollout",
        )
        if any(item in token for item in decision_hints):
            return "decision"
        if any(item in token for item in execution_hints):
            return "execution"
        return "explainer"


__all__ = ["ResearchThemeStep"]
