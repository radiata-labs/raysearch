from __future__ import annotations

import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.error_details import exception_summary, exception_to_details
from serpsage.models.errors import AppError
from serpsage.models.pipeline import ResearchQuestionCard, ResearchStepContext
from serpsage.models.research import (
    ResearchThemePlan,
    ResearchThemePlanCard,
    ThemeOutputPayload,
    ThemeQuestionCardPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    chat_pydantic,
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
        card_cap = max(1, int(self.settings.research.parallel.question_card_cap))
        seed_limit = max(6, int(ctx.runtime.budget.max_queries_per_round) * 3)
        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        payload = ThemeOutputPayload(
            detected_input_language="same as user input language",
            core_question=ctx.request.themes,
            multi_level_questions=[],
            subthemes=[],
            evidence_targets=[],
            risk_conflicts=[],
            initial_strategy="balanced",
            seed_queries=[],
            question_cards=[],
        )
        try:
            payload = await chat_pydantic(
                llm=self._llm,
                model=model,
                messages=self._build_theme_messages(
                    ctx,
                    now_utc=now_utc,
                    card_cap=card_cap,
                ),
                schema_model=ThemeOutputPayload,
                retries=int(self.settings.research.llm_self_heal_retries),
                schema_json=self._build_theme_schema(card_cap=card_cap),
            )
        except Exception as exc:  # noqa: BLE001
            summary = exception_summary(exc)
            ctx.errors.append(
                AppError(
                    code="research_theme_plan_failed",
                    message=summary,
                    details={
                        "stage": "theme_plan",
                        "model": str(model),
                        "exception": exception_to_details(exc),
                    },
                )
            )
            warnings.warn(
                (
                    "[research][warning] "
                    "research_theme_plan_failed "
                    f"request_id={ctx.request_id} "
                    f"error={summary}"
                ),
                stacklevel=1,
            )

        input_language = clean_whitespace(str(payload.detected_input_language or ""))
        if not input_language:
            input_language = "same as user input language"
        core_question = clean_whitespace(
            str(payload.core_question or ctx.request.themes)
        )
        if not core_question:
            core_question = ctx.request.themes
        multi_level_questions = normalize_strings(
            payload.multi_level_questions,
            limit=24,
        )
        subthemes = normalize_strings(payload.subthemes, limit=12)
        if not subthemes and multi_level_questions:
            subthemes = self._extract_subthemes_from_multi_level(
                multi_level_questions,
                limit=12,
            )
        seed_queries = normalize_strings(payload.seed_queries, limit=seed_limit)
        cards = self._normalize_question_cards(
            payload.question_cards,
            core_question=core_question,
            card_cap=card_cap,
            seed_limit=seed_limit,
            fallback_branches=merge_strings(
                multi_level_questions,
                subthemes,
                [core_question],
                limit=max(24, card_cap * 2),
            ),
        )

        ctx.plan.input_language = input_language
        ctx.plan.output_language = input_language
        ctx.plan.core_question = core_question
        ctx.plan.question_cards = [item.model_copy(deep=True) for item in cards]
        ctx.parallel.question_cards = [item.model_copy(deep=True) for item in cards]
        ctx.plan.theme_plan = ResearchThemePlan(
            core_question=core_question,
            multi_level_questions=multi_level_questions,
            subthemes=subthemes,
            evidence_targets=normalize_strings(payload.evidence_targets, limit=12),
            risk_conflicts=normalize_strings(payload.risk_conflicts, limit=10),
            initial_strategy=str(payload.initial_strategy or "balanced"),
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
        primary_seeds = list(cards[0].seed_queries) if cards else []
        ctx.plan.next_queries = merge_strings(
            seed_queries,
            primary_seeds,
            [core_question],
            limit=int(ctx.runtime.budget.max_queries_per_round),
        )
        ctx.corpus.coverage_state.total_subthemes = int(len(subthemes))
        ctx.notes.append(
            f"Theme plan built with {len(cards)} question cards and {len(subthemes)} subthemes."
        )
        ctx.notes.append(f"Output language fixed to {ctx.plan.output_language}.")

        print(
            (
                "[research][theme] "
                f"request_id={ctx.request_id} "
                f"core_question={ctx.plan.core_question} "
                f"question_cards={len(cards)} "
                f"subthemes={len(subthemes)} "
                f"next_queries={len(ctx.plan.next_queries)}"
            ),
            flush=True,
        )
        span.set_attr("question_cards", int(len(cards)))
        return ctx

    def _build_theme_messages(
        self,
        ctx: ResearchStepContext,
        *,
        now_utc: datetime,
        card_cap: int,
    ) -> list[dict[str, str]]:
        budget = ctx.runtime.budget
        return [
            {
                "role": "system",
                "content": (
                    "Role: Senior Research Architect.\n"
                    "Mission: Decompose THEME into executable, non-overlapping research question cards.\n"
                    "Instruction Priority:\n"
                    "P1) Schema correctness.\n"
                    "P2) Question-card execution quality.\n"
                    "P3) Decomposition power and coverage.\n"
                    "P4) Language consistency.\n"
                    "Step-by-step decomposition method:\n"
                    "1) Classify THEME into one primary type: comparison/selection, planning/how-to, diagnosis, trend/forecast, or factual mapping.\n"
                    "2) Define evidence dimensions before writing cards (for example: performance, cost, reliability, ecosystem, constraints, risk, recency).\n"
                    "3) Build a multi-level question tree (L1/L2/L3 when useful).\n"
                    "4) Convert the tree into executable question_cards, each card covering one distinct evidence objective.\n"
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
                    "8) Return JSON only.\n"
                    "Output Contract (STRICT JSON SHAPE):\n"
                    "Top-level keys allowed:\n"
                    "- detected_input_language (string)\n"
                    "- core_question (string)\n"
                    "- multi_level_questions (string[])\n"
                    "- subthemes (string[])\n"
                    "- evidence_targets (string[])\n"
                    "- risk_conflicts (string[])\n"
                    "- initial_strategy (string)\n"
                    "- seed_queries (string[])\n"
                    "- question_cards (object[])\n"
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
                    "- multi_level_questions: include explicit prefixes like L1:/L2:/L3:.\n"
                    "- question_cards: each card is one executable sub-question for one track.\n"
                    "- evidence_focus: list what evidence dimensions to prioritize.\n"
                    "- expected_gain: concrete learning value from this card.\n\n"
                    "Comparison Pattern Example (generic):\n"
                    "- Card 1: evaluate candidate A under shared criteria.\n"
                    "- Card 2: evaluate candidate B under the same criteria.\n"
                    "- Card 3: integrate evidence and decide best fit by scenario.\n\n"
                    "JSON Skeleton Example:\n"
                    "{\n"
                    '  "detected_input_language": "zh",\n'
                    '  "core_question": "...",\n'
                    '  "multi_level_questions": [],\n'
                    '  "subthemes": [],\n'
                    '  "evidence_targets": [],\n'
                    '  "risk_conflicts": [],\n'
                    '  "initial_strategy": "balanced",\n'
                    '  "seed_queries": [],\n'
                    '  "question_cards": [\n'
                    "    {\n"
                    '      "question": "...",\n'
                    '      "priority": 3,\n'
                    '      "seed_queries": ["..."],\n'
                    '      "evidence_focus": ["..."],\n'
                    '      "expected_gain": "..."\n'
                    "    }\n"
                    "  ]\n"
                    "}"
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
                "multi_level_questions",
                "subthemes",
                "evidence_targets",
                "risk_conflicts",
                "initial_strategy",
                "seed_queries",
                "question_cards",
            ],
            "properties": {
                "detected_input_language": {"type": "string"},
                "core_question": {"type": "string"},
                "multi_level_questions": {
                    "type": "array",
                    "maxItems": 24,
                    "items": {"type": "string"},
                },
                "subthemes": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string"},
                },
                "evidence_targets": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string"},
                },
                "risk_conflicts": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {"type": "string"},
                },
                "initial_strategy": {"type": "string"},
                "seed_queries": {
                    "type": "array",
                    "maxItems": 12,
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
                        [question],
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
