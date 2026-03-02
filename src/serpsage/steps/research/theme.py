from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.pipeline import ResearchQuestionCard, ResearchStepContext
from serpsage.models.research import (
    ReportStyle,
    ResearchThemePlan,
    ResearchThemePlanCard,
    ThemeOutputPayload,
    ThemeQuestionCardPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt_style import (
    UNIVERSAL_GUARDRAILS,
    build_style_overlay,
    compose_system_prompt,
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
        card_cap = max(1, int(self.settings.research.parallel.question_card_cap))
        seed_limit = max(6, int(ctx.runtime.budget.max_queries_per_round) * 3)
        style_cfg = self.settings.research.report_style
        hinted_style = infer_report_style_from_theme(
            ctx.request.themes,
            default=self._normalize_style_fallback(style_cfg.fallback_style),
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
                    card_cap=card_cap,
                ),
                response_format=ThemeOutputPayload,
                format_override=self._build_theme_schema(card_cap=card_cap),
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
                "report_style_selected": str(report_style),
                "report_style_fallback_used": bool(style_fallback_used),
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
        hinted_style = infer_report_style_from_theme(
            ctx.request.themes,
            default=self._normalize_style_fallback(
                self.settings.research.report_style.fallback_style
            ),
        )
        system_contract = (
            "Role: Senior Research Architect.\n"
            "Mission: Decompose THEME into executable, non-overlapping research question cards and classify one report_style.\n"
            "Instruction Priority:\n"
            "P1) Schema correctness.\n"
            "P2) Question-card execution quality.\n"
            "P3) Decomposition power and coverage.\n"
            "P4) Report-style fit for user task intent.\n"
            "P5) Language consistency.\n"
            "Step-by-step decomposition method:\n"
            "1) Classify THEME into one primary type: comparison/selection, planning/how-to, diagnosis, trend/forecast, or factual mapping.\n"
            "2) Predict best report_style for user value: decision/explainer/execution.\n"
            "3) Define evidence dimensions before writing cards (for example: performance, cost, reliability, ecosystem, constraints, risk, recency).\n"
            "4) Identify subthemes that ensure high coverage with low overlap.\n"
            "5) Convert subthemes into executable question_cards, each card covering one distinct evidence objective.\n"
            "Question-type playbook:\n"
            "A) Comparison / Selection question:\n"
            "- If THEME compares N candidates, create candidate cards (one per candidate) and one synthesis card.\n"
            "- Candidate cards: evaluate one candidate only using shared criteria.\n"
            "- Synthesis card: answer which option fits which scenario and why.\n"
            "- For two-candidate comparisons, target structure is: candidate A card + candidate B card + final recommendation card.\n"
            "B) Planning / How-to question:\n"
            "- Split into goal definition, implementation path, bottlenecks, and validation criteria.\n"
            "C) Diagnosis / Why question:\n"
            "- Split into symptom framing, root-cause hypotheses, disambiguation evidence, and fix validation.\n"
            "D) Trend / Forecast question:\n"
            "- Split into current baseline, drivers, constraints, and forward-looking scenarios with time anchors.\n"
            "Hard Constraints:\n"
            "1) Output free-text in detected_input_language.\n"
            "2) Every question card must be externally verifiable by web evidence.\n"
            "3) question_cards must be deduplicated and practical for a single track loop.\n"
            "4) Each card must focus on one sub-problem, not the entire THEME restated.\n"
            "5) Each card needs distinct evidence_focus and high-yield seed_queries.\n"
            "6) priority must be 1..5 (5 highest).\n"
            f"7) Return at most {card_cap} question cards.\n"
            "8) Do not return top-level seed_queries; seed queries must be inside question_cards items.\n"
            "9) report_style must be exactly one of: decision, explainer, execution.\n"
            "10) Return JSON only.\n"
            "Output Contract (STRICT JSON SHAPE):\n"
            "Top-level keys allowed:\n"
            "- detected_input_language (string)\n"
            "- core_question (string)\n"
            "- report_style (decision|explainer|execution)\n"
            "- subthemes (string[])\n"
            "- required_entities (string[])\n"
            "- question_cards (object[])\n"
            "required_entities policy:\n"
            "- If THEME compares/evaluates named entities, include each entity as an exact surface form.\n"
            "- Keep versions/suffixes intact (for example qwen3.5, glm4.7, llama-3.1).\n"
            "- If no concrete named entities are required, return an empty array.\n"
            "question_cards item keys allowed:\n"
            "- question (string)\n"
            "- priority (integer 1..5)\n"
            "- seed_queries (string[])\n"
            "- evidence_focus (string[])\n"
            "- expected_gain (string)\n"
            "Do not add question_id in question_cards; question_id is generated by runtime.\n"
            "Quality Checklist:\n"
            "- High coverage, low overlap, concrete queries, explicit expected gain.\n"
            "- Every question_cards item must include question, priority, seed_queries, evidence_focus, expected_gain.\n"
            "- Comparison questions must include per-candidate cards and one synthesis/final-decision card."
        )
        return [
            {
                "role": "system",
                "content": compose_system_prompt(
                    base_contract=system_contract,
                    style_overlay=build_style_overlay(
                        stage="theme",
                        style=hinted_style,
                    ),
                    universal_guardrails=UNIVERSAL_GUARDRAILS,
                ),
            },
            {
                "role": "user",
                "content": (
                    f"THEME:\n{ctx.request.themes}\n\n"
                    f"SEARCH_MODE:\n{ctx.request.search_mode}\n\n"
                    "TIME_CONTEXT:\n"
                    f"- current_utc_timestamp={now_utc.isoformat()}\n"
                    f"- current_utc_date={now_utc.date().isoformat()}\n\n"
                    "TEMPORAL_POLICY:\n"
                    "- If THEME asks for latest/current/today/now/as of, include explicit time anchors in seed queries.\n"
                    "- Resolve relative words against current_utc_date.\n\n"
                    "BUDGET_HINTS:\n"
                    f"- max_rounds={budget.max_rounds}\n"
                    f"- max_search_calls={budget.max_search_calls}\n"
                    f"- max_queries_per_round={budget.max_queries_per_round}\n\n"
                    "DECOMPOSITION_POLICY:\n"
                    "- Prefer fewer high-information cards over many generic cards.\n"
                    "- Avoid cards that only rephrase THEME.\n"
                    "- Make evidence_focus mutually informative (little overlap).\n"
                    "- For comparison themes, include candidate-specific cards and one final synthesis card.\n\n"
                    "Output Notes:\n"
                    "- core_question: one-sentence anchor question.\n"
                    "- report_style: classify best user-facing report style as decision/explainer/execution.\n"
                    "- required_entities: exact strings that must be covered by evidence and later summaries.\n"
                    "- question_cards: each card is one executable sub-question for one track.\n"
                    "- Do not produce top-level seed_queries.\n"
                    "- evidence_focus: list what evidence dimensions to prioritize.\n"
                    "- expected_gain: concrete learning value from this card.\n\n"
                    "Comparison Pattern Example (generic):\n"
                    "- Card 1: evaluate candidate A under shared criteria.\n"
                    "- Card 2: evaluate candidate B under the same criteria.\n"
                    "- Card 3: integrate evidence and decide best fit by scenario."
                ),
            },
        ]

    def _build_theme_schema(self, *, card_cap: int) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "detected_input_language",
                "core_question",
                "report_style",
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


__all__ = ["ResearchThemeStep"]
