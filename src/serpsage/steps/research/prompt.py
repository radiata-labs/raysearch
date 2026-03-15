from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from serpsage.models.steps.research import (
    OverviewOutputPayload,
    RenderArchitectOutput,
    RenderArchitectSectionPlan,
    ReportStyle,
    ResearchLinkCandidate,
    ResearchQuestionCard,
    ResearchRound,
    ResearchSource,
    ResearchStepContext,
    ResearchTask,
    ResearchThemePlan,
    ResearchThemePlanCard,
    ResearchTrackResult,
    TaskComplexity,
    TaskIntent,
)
from serpsage.models.steps.search import QuerySourceSpec

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

PromptStage = Literal[
    "theme",
    "plan",
    "overview",
    "content",
    "subreport",
    "render_architect",
    "render_writer",
    "render_structured",
]

_STYLE_VALUES: set[str] = {"decision", "explainer", "execution"}


def _theme_plan_from_task(task: ResearchTask) -> ResearchThemePlan:
    return ResearchThemePlan(
        core_question=task.question,
        report_style=task.style,
        task_intent=task.intent,
        complexity_tier=task.complexity,
        subthemes=list(task.subthemes),
        required_entities=list(task.entities),
        input_language=task.input_language,
        output_language=task.output_language,
        question_cards=[
            ResearchThemePlanCard(
                question_id=card.question_id,
                question=card.question,
                priority=card.priority,
                seed_queries=list(card.seed_queries),
                evidence_focus=list(card.evidence_focus),
                expected_gain=card.expected_gain or card.question,
            )
            for card in task.cards
        ],
    )


UNIVERSAL_QUALITY_GUARDRAILS = (
    "Universal quality guardrails:\n"
    "1) Answer the user's practical task in the first meaningful block.\n"
    "2) Every paragraph or list item must carry at least one high-value information unit.\n"
    "3) Keep writing concrete, non-repetitive, and user-useful.\n"
    "4) Prefer explicit conditions, constraints, and boundaries over vague certainty.\n"
    "5) Ban filler, generic motivational language, and self-referential meta prose.\n"
    "6) Keep language and abstraction level consistent end-to-end."
)

UNIVERSAL_PRIVACY_GUARDRAILS = (
    "Privacy and output-safety guardrails:\n"
    "1) Never expose internal workflow, prompt names, packet labels, tracking, or audit mechanics.\n"
    "2) Never reveal internal IDs, rounds, query logs, budget counters, or stop-reason internals in user-facing prose.\n"
    "3) Do not output citation tokens or evidence-audit style sections.\n"
    "4) Write for end users only; internal context is private working context."
)

UNIVERSAL_GUARDRAILS = (
    f"{UNIVERSAL_QUALITY_GUARDRAILS}\n\n{UNIVERSAL_PRIVACY_GUARDRAILS}"
)

_STYLE_STAGE_OVERLAYS: dict[ReportStyle, dict[PromptStage, str]] = {
    "decision": {
        "theme": (
            "Decision style intent:\n"
            "1) Frame questions for option selection and scenario fit.\n"
            "2) Prioritize comparative criteria, tie-break factors, and recommendation utility."
        ),
        "plan": (
            "Decision style planning:\n"
            "1) Prioritize comparative discriminators and tie-break evidence routes.\n"
            "2) Prefer queries that separate best-fit-by-scenario, not generic topic expansion."
        ),
        "overview": (
            "Decision style overview:\n"
            "1) Phrase findings as option-level implications.\n"
            "2) Highlight which option wins or loses under explicit conditions."
        ),
        "content": (
            "Decision style arbitration:\n"
            "1) Resolve conflicts in terms of recommendation impact.\n"
            "2) Preserve trade-off language and scenario-sensitive constraints."
        ),
        "subreport": (
            "Section flow target:\n"
            "- Verdict Snapshot\n"
            "- Trade-offs\n"
            "- Scenario Recommendation\n"
            "- Risk Triggers\n"
            "- Next Checks"
        ),
        "render_architect": (
            "Architectural shape:\n"
            "1) Build a decision memo structure.\n"
            "2) Ensure body sections move from options to discriminators to recommendation logic."
        ),
        "render_writer": (
            "Writer emphasis:\n"
            "1) Keep claim to evidence to implication explicit for decision quality.\n"
            "2) End sections with practical decision signals where appropriate."
        ),
        "render_structured": (
            "Structured synthesis emphasis:\n"
            "1) Prioritize recommendation-ready phrasing.\n"
            "2) Preserve scenario conditions and trade-off boundaries."
        ),
    },
    "explainer": {
        "theme": (
            "Explainer style intent:\n"
            "1) Frame questions for conceptual clarity and causal understanding.\n"
            "2) Prioritize mechanism-level decomposition and boundary conditions."
        ),
        "plan": (
            "Explainer style planning:\n"
            "1) Prioritize conceptual breadth and mechanism clarity.\n"
            "2) Prefer queries that reduce confusion and unify fragmented explanations."
        ),
        "overview": (
            "Explainer style overview:\n"
            "1) Phrase findings around causal and structural understanding.\n"
            "2) Surface misconceptions, limits, and boundary cases."
        ),
        "content": (
            "Explainer style arbitration:\n"
            "1) Resolve conflicts by clarifying conceptual mechanisms.\n"
            "2) Highlight where definitions or assumptions diverge."
        ),
        "subreport": (
            "Section flow target:\n"
            "- Core Model\n"
            "- Mechanisms\n"
            "- Boundary Cases\n"
            "- Common Misconceptions\n"
            "- Practical Takeaway"
        ),
        "render_architect": (
            "Architectural shape:\n"
            "1) Build a teaching-oriented narrative.\n"
            "2) Sequence sections from fundamentals to nuanced constraints."
        ),
        "render_writer": (
            "Writer emphasis:\n"
            "1) Explain why, not only what.\n"
            "2) Keep abstraction controlled and grounded in concrete examples."
        ),
        "render_structured": (
            "Structured synthesis emphasis:\n"
            "1) Maximize conceptual clarity and causal coherence.\n"
            "2) Preserve limits and exception cases."
        ),
    },
    "execution": {
        "theme": (
            "Execution style intent:\n"
            "1) Frame questions for implementation readiness and delivery risk.\n"
            "2) Prioritize prerequisites, constraints, procedures, and validation steps."
        ),
        "plan": (
            "Execution style planning:\n"
            "1) Prioritize operational prerequisites, dependency checks, and failure-prone steps.\n"
            "2) Prefer queries that produce implementable actions and controls."
        ),
        "overview": (
            "Execution style overview:\n"
            "1) Phrase findings as actionable blockers and readiness signals.\n"
            "2) Distinguish mandatory steps from optional optimizations."
        ),
        "content": (
            "Execution style arbitration:\n"
            "1) Resolve conflicts by deployment and operations impact.\n"
            "2) Preserve procedural risk and control language."
        ),
        "subreport": (
            "Section flow target:\n"
            "- Goal and Prerequisites\n"
            "- Step Sequence\n"
            "- Validation Criteria\n"
            "- Failure Handling\n"
            "- Next Actions"
        ),
        "render_architect": (
            "Architectural shape:\n"
            "1) Build an operations playbook structure.\n"
            "2) Ensure section order supports execution from setup to verification."
        ),
        "render_writer": (
            "Writer emphasis:\n"
            "1) Write implementation-ready guidance with explicit constraints.\n"
            "2) Include checkpoints and failure contingencies where relevant."
        ),
        "render_structured": (
            "Structured synthesis emphasis:\n"
            "1) Optimize for actionability and execution sequencing.\n"
            "2) Preserve validation and fallback conditions."
        ),
    },
}


def _build_style_overlay(*, stage: PromptStage, style: ReportStyle) -> str:
    normalized_style = _normalize_style(style) or "explainer"
    out = _STYLE_STAGE_OVERLAYS.get(normalized_style, {}).get(stage, "")
    return out.strip()


def _compose_system_prompt(
    *,
    base_contract: str,
    style_overlay: str,
    universal_guardrails: str,
) -> str:
    base = base_contract.strip()
    overlay = style_overlay.strip()
    guardrails = universal_guardrails.strip()
    parts = [item for item in [guardrails, base] if item]
    if overlay:
        parts.append(f"Style overlay:\n{overlay}")
    return "\n\n".join(parts).strip()


def _normalize_mode_name(mode_key: str) -> str:
    return mode_key.strip().casefold()


def _theme_depth_contract(*, mode_key: str) -> str:
    mode_name = _normalize_mode_name(mode_key)
    if mode_name == "research-fast":
        return (
            "- fast mode: converge quickly with a compact question-card set.\n"
            "- prioritize immediate answerability and low-latency coverage."
        )
    if mode_name == "research-pro":
        return (
            "- pro mode: decompose deeply with boundary conditions, failure scenarios, and tie-break discriminators.\n"
            "- for simple how-to tasks, enforce shortest executable path first, then expand to risk/governance only if core path is complete.\n"
            "- include high-yield cards that unlock large downstream information gain."
        )
    return (
        "- research mode: balance broad coverage with comparative depth.\n"
        "- produce enough cards to avoid blind spots while keeping low overlap."
    )


def _mode_depth_planning_contract(*, mode_key: str) -> str:
    mode_name = _normalize_mode_name(mode_key)
    if mode_name == "research-fast":
        return (
            "- fast: prioritize rapid convergence with 1-2 high-yield query intents.\n"
            "- avoid broad exploration when likely gain is low."
        )
    if mode_name == "research-pro":
        return (
            "- pro: include deeper discriminators, boundary-case routes, and failure-mode checks.\n"
            "- for simple how-to tasks, prioritize quick-start and critical step sequence queries before governance/cost extensions.\n"
            "- prioritize queries that unlock multiple downstream sections in one pass."
        )
    return (
        "- research: balance coverage and depth, ensuring each query contributes a distinct information slot.\n"
        "- research: prioritize unresolved core-answer slots before optional side expansion.\n"
        "- include at least one verification-oriented query when conflicts remain."
    )


def research_mode_scope_lock_contract(
    *,
    mode_key: str,
    task_intent: TaskIntent,
) -> str:
    mode_name = _normalize_mode_name(mode_key)
    if mode_name != "research":
        return ""
    if task_intent == "how_to":
        return (
            "Research-mode scope lock:\n"
            "1) First body section must provide the shortest executable path.\n"
            "2) Second body section must cover critical steps and failure-prevention checks.\n"
            "3) Optional expansion is allowed only after the core path is fully covered and may appear at most once."
        )
    return (
        "Research-mode scope lock:\n"
        "1) First meaningful block must directly answer the core user task.\n"
        "2) Prioritize unresolved core question_ids before any optional expansion.\n"
        "3) Optional expansion is allowed only after core coverage and may appear at most once."
    )


def _resolve_language_label(language_code: str) -> str:
    normalized = language_code.strip() if language_code else ""
    return normalized or "the required language"


def _build_language_lock_block(
    *,
    field_name: str,
    language_code: str,
    language_label: str,
) -> str:
    return (
        f"{field_name}_LABEL:\n{language_code} ({language_label})\n\n"
        f"{field_name}_LOCK:\n"
        f"- {field_name} is {language_code} ({language_label}).\n"
        f"- Write all user-facing prose and every free-text string directly in {language_label}.\n"
        f"- Do not answer in English unless {field_name} is en (English).\n"
    )


def _question_card_field_contract() -> str:
    return (
        "Field definitions and scoring rubrics:\n"
        "- core_question: one sentence only; capture the user's true decision or understanding target, not background chatter.\n"
        "- subthemes: 3-8 short coverage dimensions when possible; each must describe a distinct evidence slot, not a synonym of another item.\n"
        "- required_entities: include exact names/versions only when later research must explicitly cover them; otherwise return [].\n"
        "- priority scale: 5=core blocker or highest expected information gain, 4=important tie-break or major risk, 3=useful supporting dimension, 2=secondary nuance, 1=optional polish.\n"
        "- question: answerable with web evidence in one isolated card loop; avoid multi-part bundles joined by 'and'.\n"
        "- seed_queries: 1-4 preferred, 8 max; each query should open a distinct retrieval route rather than trivial wording variants.\n"
        "- evidence_focus: list concrete evidence dimensions or discriminators such as benchmark, pricing, policy, compatibility, recency, or failure mode.\n"
        "- expected_gain: describe the concrete user value unlocked by this card, such as eliminating a tie-break, validating a key risk, or clarifying a mechanism.\n"
        "- Comparison tasks: include candidate-specific cards plus one synthesis card unless the request is clearly single-option.\n"
    )


def _plan_field_contract() -> str:
    return (
        "Field definitions and quantitative planning rubrics:\n"
        "- query_strategy: 1-2 sentences stating what uncertainty this round targets and why it is highest-yield now.\n"
        "- round_action: choose explore only when last-round links are clearly more promising than a fresh search; otherwise choose search.\n"
        "- search_jobs: order by expected information gain; the first job should be the one you would keep if only one executes.\n"
        "- intent rubric: coverage=fill missing subtheme/entity coverage, deepen=expand a promising evidence branch, verify=resolve contradiction or tie-break, refresh=check current/latest state.\n"
        "- mode rubric: auto for standard retrieval, deep when broad recall, contradiction checking, or authoritative corroboration matters.\n"
        "- query: one concrete query string, not a prose explanation.\n"
        "- additional_queries: use for adjacent retrieval routes; avoid spelling-only variants or redundant rephrasings.\n"
        "- De-duplication rule: if two jobs would likely return the same top results, keep the sharper one and delete the weaker one.\n"
    )


def _overview_field_contract() -> str:
    return (
        "Field definitions and calibration rules:\n"
        "- findings: each item should compress one evidence-backed point into conclusion -> condition/boundary -> impact.\n"
        "- conflict_arbitration.status: resolved only when overview evidence already gives a credible direction; otherwise unresolved.\n"
        "- covered_subthemes: include only subthemes with meaningful evidence, not merely mentioned topics.\n"
        "- entity_coverage_complete=true only when every required entity has non-trivial evidence support.\n"
        "- critical_gaps: include only answer-blocking gaps, not minor nice-to-have follow-ups.\n"
        "- confidence scale: 0.8 to 1.0=strong and well-supported, 0.4 to 0.79=useful but incomplete, 0.0 to 0.39=weak/mixed, below 0=conflicted, stale, or likely misleading.\n"
        "- need_content_source_ids: choose sources that are high-impact, contradictory, authority-rich, or too important to judge from overview alone.\n"
        "- next_query_strategy: one short rationale for the next retrieval move.\n"
        "- next_queries: 0-4 preferred, ordered by gain; each query must reduce a concrete gap.\n"
    )


def _content_field_contract() -> str:
    return (
        "Field definitions and calibration rules:\n"
        "- resolved_findings: each item should compress one full-content conclusion into conclusion -> condition/boundary -> impact.\n"
        "- conflict_resolutions.status: resolved=clear winner supported by content, unresolved=real tie remains, insufficient=not enough evidence, closed=topic no longer matters after review.\n"
        "- entity_coverage_complete=true only when every required entity has enough full-context evidence for the current card.\n"
        "- remaining_gaps: include only gaps that still materially weaken the answer.\n"
        "- confidence_adjustment scale: +0.4 to +1.0 for major strengthening evidence, +0.1 to +0.39 for moderate strengthening, around 0 for little change, -0.1 to -0.39 for meaningful weakening, below -0.4 for severe contradiction or staleness.\n"
        "- next_query_strategy: one short rationale for what evidence should be pursued next, if any.\n"
        "- next_queries: 0-4 preferred, de-duplicated, and tightly scoped to unresolved gaps.\n"
    )


def _track_insight_card_contract() -> str:
    return (
        "Track insight card definitions:\n"
        "- direct_answer: one compact answer sentence for the card's question.\n"
        "- high_value_points: each point must include conclusion, condition, and impact; avoid generic summaries.\n"
        "- key_tradeoffs_or_mechanisms: list the levers, mechanisms, or trade-offs that explain why the answer looks this way.\n"
        "- unknowns_and_risks: include only material unresolved risks or uncertainty boundaries.\n"
        "- next_actions: concrete user actions or checks, not abstract advice.\n"
    )


def _build_theme_messages(
    *,
    theme: str,
    search_mode: str,
    mode_depth_profile: str,
    current_utc_timestamp: str,
    current_utc_date: str,
    max_rounds: int,
    max_search_calls: int,
    max_queries_per_round: int,
    card_cap: int,
    hinted_style: ReportStyle,
) -> list[dict[str, str]]:
    depth_contract = _theme_depth_contract(mode_key=mode_depth_profile)
    system_contract = (  # noqa: S608
        "Role: Senior Research Architect.\n"
        "Mission: Decompose THEME into executable, non-overlapping research question cards and classify report_style, task_intent, complexity_tier, and detected_input_language.\n"
        "Instruction Priority:\n"
        "P1) Schema correctness.\n"
        "P2) Question-card execution quality.\n"
        "P3) Decomposition power and coverage.\n"
        "P4) Intent/complexity classification fit for user task.\n"
        "P5) Report-style fit for user task intent.\n"
        "P6) Language classification quality.\n"
        "Step-by-step decomposition method:\n"
        "1) Classify THEME into one primary type: comparison/selection, planning/how-to, diagnosis, trend/forecast, or factual mapping.\n"
        "2) Predict task_intent as one of how_to/comparison/explainer/diagnosis/other.\n"
        "3) Predict complexity_tier as one of low/medium/high based on user task complexity.\n"
        "4) Predict best report_style for user value: decision/explainer/execution.\n"
        "5) Predict detected_input_language as canonical code (for example zh-Hans, zh-Hant, en, ja, ko, fr, de).\n"
        "6) Define evidence dimensions before writing cards (for example: performance, cost, reliability, ecosystem, constraints, risk, recency).\n"
        "7) Identify subthemes that ensure high coverage with low overlap.\n"
        "8) Convert subthemes into executable question_cards, each card covering one distinct evidence objective.\n"
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
        "1) Output free-text in detected_input_language; this means the actual prose itself, not a language tag.\n"
        "2) Every question card must be externally verifiable by web evidence.\n"
        "3) question_cards must be deduplicated and practical for a single track loop.\n"
        "4) Each card must focus on one sub-problem, not the entire THEME restated.\n"
        "5) Each card needs distinct evidence_focus and high-yield seed_queries.\n"
        "6) priority must be 1..5 (5 highest).\n"
        f"7) Return at most {card_cap} question cards.\n"
        "8) Do not return top-level seed_queries; seed queries must be inside question_cards items.\n"
        "9) report_style must be exactly one of: decision, explainer, execution.\n"
        "10) task_intent must be exactly one of: how_to, comparison, explainer, diagnosis, other.\n"
        "11) complexity_tier must be exactly one of: low, medium, high.\n"
        "12) detected_input_language must be canonical language code.\n"
        "13) Return JSON only.\n"
        "Output Contract (STRICT JSON SHAPE):\n"
        "Top-level keys allowed:\n"
        "- detected_input_language (canonical language code string)\n"
        "- core_question (string)\n"
        "- report_style (decision|explainer|execution)\n"
        "- task_intent (how_to|comparison|explainer|diagnosis|other)\n"
        "- complexity_tier (low|medium|high)\n"
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
        "- Comparison questions must include per-candidate cards and one synthesis/final-decision card.\n"
        f"{_question_card_field_contract()}"
        "Mode-depth contract:\n"
        f"{depth_contract}"
    )
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=_build_style_overlay(stage="theme", style=hinted_style),
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"THEME:\n{theme}\n\n"
                f"SEARCH_MODE:\n{search_mode}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                "TIME_CONTEXT:\n"
                f"- current_utc_timestamp={current_utc_timestamp}\n"
                f"- current_utc_date={current_utc_date}\n\n"
                "TEMPORAL_POLICY:\n"
                "- If THEME asks for latest/current/today/now/as of, include explicit time anchors in seed queries.\n"
                "- Resolve relative words against current_utc_date.\n\n"
                "BUDGET_HINTS:\n"
                f"- max_rounds={max_rounds}\n"
                f"- max_search_calls={max_search_calls}\n"
                f"- max_queries_per_round={max_queries_per_round}\n\n"
                "DECOMPOSITION_POLICY:\n"
                "- Prefer fewer high-information cards over many generic cards.\n"
                "- Avoid cards that only rephrase THEME.\n"
                "- Make evidence_focus mutually informative (little overlap).\n"
                "- For comparison themes, include candidate-specific cards and one final synthesis card.\n\n"
                "Output Notes:\n"
                "LANGUAGE_POLICY:\n"
                "- Preserve the writing language of THEME in every free-text field.\n"
                "- If THEME is Chinese, write free-text in Chinese characters instead of English paraphrases.\n\n"
                "- core_question: one-sentence anchor question.\n"
                "- report_style: classify best user-facing report style as decision/explainer/execution.\n"
                "- task_intent: classify user task intent as how_to/comparison/explainer/diagnosis/other.\n"
                "- complexity_tier: classify task complexity as low/medium/high.\n"
                "- detected_input_language: canonical language code for user question language.\n"
                "- required_entities: exact strings that must be covered by evidence and later summaries.\n"
                "- question_cards: each card is one executable sub-question for one track.\n"
                "- Do not produce top-level seed_queries.\n"
                "- evidence_focus: list what evidence dimensions to prioritize.\n"
                "- expected_gain: concrete learning value from this card.\n\n"
                f"{_question_card_field_contract()}\n"
                "Comparison Pattern Example (generic):\n"
                "- Card 1: evaluate candidate A under shared criteria.\n"
                "- Card 2: evaluate candidate B under the same criteria.\n"
                "- Card 3: integrate evidence and decide best fit by scenario."
            ),
        },
    ]


def _build_plan_messages(
    *,
    theme: str,
    core_question: str,
    report_style: ReportStyle,
    mode_depth_profile: str,
    round_index: int,
    current_utc_timestamp: str,
    current_utc_date: str,
    required_output_language: str,
    required_output_language_label: str,
    theme_plan_markdown: str,
    previous_rounds_markdown: str,
    candidate_queries_markdown: str,
    required_entities: list[str],
    search_calls_remaining: int,
    fetch_calls_remaining: int,
    max_queries_this_round: int,
    allow_explore: bool,
    explore_target_pages_per_round: int,
    explore_links_per_page: int,
    last_round_link_candidates_markdown: str,
) -> list[dict[str, str]]:
    depth_contract = _mode_depth_planning_contract(mode_key=mode_depth_profile)
    system_contract = (  # noqa: S608
        "Role: Principal Research Planner.\n"  # noqa: S608
        "Mission: Select the next-round action and produce focused search jobs.\n"
        "Instruction Priority:\n"
        "P1) Schema correctness and budget adherence.\n"
        "P2) Action fit (search vs explore) for CORE_QUESTION.\n"
        "P3) Information gain per query.\n"
        "P4) Output language consistency.\n"
        "Hard Constraints:\n"
        "1) Return JSON with exactly: query_strategy, round_action, explore_target_source_ids, search_jobs.\n"
        "2) round_action must be exactly search or explore.\n"
        "3) Round 1 must use round_action=search.\n"
        "4) If explore is not allowed in this round, round_action must be search.\n"
        "5) If round_action=explore, select only source IDs from LAST_ROUND_LINK_CANDIDATES and keep list length <= explore_target_pages_per_round.\n"
        "6) If round_action=search, explore_target_source_ids must be an empty array.\n"
        "7) Always return a valid search_jobs array when search budget remains, even if round_action=explore.\n"
        "8) All search_jobs must serve CORE_QUESTION. Do not introduce a new independent research question.\n"
        "9) Minimize overlap between jobs while maximizing information gain.\n"
        "10) Respect remaining budget and prioritize unresolved high-value gaps.\n"
        "11) Use deep mode when higher recall or conflict verification is needed.\n"
        "12) Free-text fields must be in required_output_language, but search queries may use the language best suited for recall.\n"
        "13) Temporal grounding: interpret relative time words against current UTC date.\n"
        "14) If recency intent exists, include explicit temporal constraints in query text.\n"
        "15) For high-impact claims, prioritize authoritative routes (official documentation, primary sources, standards, vendor announcements, peer-reviewed or institution-backed reports).\n"
        "16) Preserve required_entities exact surface forms and version strings inside query/additional_queries.\n"
        "17) When max_queries_this_round is 1, prefer one deep job with additional_queries for breadth.\n"
        "18) If focus cannot be preserved, return fewer search_jobs or an empty array; never drift off-topic.\n"
        "19) Return JSON only; no markdown or explanations.\n"
        "Allowed Inputs:\n"
        "- User theme, theme plan, previous round summaries, candidate queries, and last-round link candidates.\n"
        "Failure Policy:\n"
        "- If uncertain about explore value, choose search.\n"
        "- If uncertain, produce fewer but higher-value jobs.\n"
        "- If all candidate queries are off-topic relative to CORE_QUESTION, return search_jobs as an empty array.\n"
        "Quality Checklist:\n"
        "- Action choice is justified by expected information gain.\n"
        "- Distinct intent per job, no near duplicates, conflict-aware targeting.\n"
        f"{_plan_field_contract()}"
        "Mode-depth contract:\n"
        f"{depth_contract}"
    )
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=_build_style_overlay(stage="plan", style=report_style),
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"THEME:\n{theme}\n\n"
                f"CORE_QUESTION:\n{core_question}\n\n"
                f"REPORT_STYLE_LOCKED:\n{report_style}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"ROUND_INDEX:\n{round_index}\n\n"
                "TIME_CONTEXT:\n"
                f"- current_utc_timestamp={current_utc_timestamp}\n"
                f"- current_utc_date={current_utc_date}\n\n"
                "TEMPORAL_POLICY:\n"
                "- If theme/round context asks for latest/current/today/now/as of/this year/month/week, queries must include concrete temporal anchors.\n"
                "- When no recency intent is present, avoid over-constraining by date.\n"
                "- Prefer fresh/authoritative sources for recency queries.\n\n"
                "LANGUAGE_POLICY:\n"
                f"{_build_language_lock_block(field_name='required_output_language', language_code=required_output_language, language_label=required_output_language_label)}\n"
                "- Keep planner prose in required_output_language.\n"
                "- Use whichever query language maximizes recall for the current card.\n\n"
                "ROUND_ACTION_POLICY:\n"
                "- round 1: must output round_action=search.\n"
                "- round > 1: may choose round_action=explore only when allow_explore=true.\n"
                f"- allow_explore_this_round={str(bool(allow_explore)).lower()}\n"
                f"- explore_target_pages_per_round={explore_target_pages_per_round}\n"
                f"- explore_links_per_page={explore_links_per_page}\n\n"
                f"THEME_PLAN_MARKDOWN:\n{theme_plan_markdown}\n\n"
                f"PREVIOUS_ROUNDS_MARKDOWN:\n{previous_rounds_markdown}\n\n"
                f"CANDIDATE_QUERIES_MARKDOWN:\n{candidate_queries_markdown}\n\n"
                "LAST_ROUND_LINK_CANDIDATES:\n"
                f"{last_round_link_candidates_markdown}\n\n"
                "ENTITY_POLICY:\n"
                f"- required_entities={required_entities}\n"
                "- Every required entity must appear in query or additional_queries as exact text.\n"
                "- Do not drop version markers such as dots/hyphens (for example qwen3.5, glm4.7, llama-3.1).\n\n"
                "BUDGET_REMAINING:\n"
                f"- search_calls_remaining={search_calls_remaining}\n"
                f"- fetch_calls_remaining={fetch_calls_remaining}\n"
                f"- max_queries_this_round={max_queries_this_round}\n\n"
                "Job design rubric:\n"
                "- coverage: close missing subtheme coverage.\n"
                "- deepen: improve depth on high-value evidence branches.\n"
                "- verify: target contradiction resolution and tie-break evidence.\n"
                "- refresh: prioritize latest authoritative updates.\n"
                "- if max_queries_this_round=1, prefer deep mode with additional_queries for one-call breadth.\n\n"
                f"{_plan_field_contract()}\n"
                "OUTPUT_SHAPE_HINT:\n"
                "{\n"
                '  "query_strategy": "...",\n'
                '  "round_action": "search|explore",\n'
                '  "explore_target_source_ids": [1,2],\n'
                '  "search_jobs": [{ "query": "..." }]\n'
                "}"
            ),
        },
    ]


def _build_link_picker_messages(
    *,
    core_question: str,
    report_style: ReportStyle,
    mode_depth_profile: str,
    current_utc_date: str,
    source_id: int,
    source_url: str,
    source_title: str,
    max_links_to_select: int,
    candidate_links_markdown: str,
) -> list[dict[str, str]]:
    system_contract = (  # noqa: S608
        "Role: Link Selection Analyst.\n"
        "Mission: Select the highest-yield deep-explore links for one source page.\n"
        "Hard Constraints:\n"
        "1) Use both link text and URL when scoring relevance and authority.\n"
        "2) Prioritize official docs/specs, standards, papers/preprints, repositories, and technical references.\n"
        "3) De-prioritize navigation pages, login/signup, pricing, privacy/terms, marketing, and generic index pages.\n"
        "4) Keep selection strictly scoped to CORE_QUESTION.\n"
        "5) Select at most max_links_to_select IDs.\n"
        "6) Return JSON only: selected_link_ids.\n"
        "7) selected_link_ids must reference IDs from CANDIDATE_LINKS only.\n"
        "8) If no candidate is useful, return selected_link_ids as an empty array.\n"
    )
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=_build_style_overlay(stage="plan", style=report_style),
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"CORE_QUESTION:\n{core_question}\n\n"
                f"REPORT_STYLE_LOCKED:\n{report_style}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"CURRENT_UTC_DATE:\n{current_utc_date}\n\n"
                "SOURCE_CONTEXT:\n"
                f"- source_id={int(source_id)}\n"
                f"- source_url={source_url}\n"
                f"- source_title={source_title}\n\n"
                f"MAX_LINKS_TO_SELECT:\n{int(max_links_to_select)}\n\n"
                "CANDIDATE_LINKS:\n"
                f"{candidate_links_markdown}\n\n"
                "Output JSON shape:\n"
                "{\n"
                '  "selected_link_ids": [1,2]\n'
                "}"
            ),
        },
    ]


def _build_overview_messages(
    *,
    theme: str,
    core_question: str,
    report_style: ReportStyle,
    mode_depth_profile: str,
    round_index: int,
    current_utc_timestamp: str,
    current_utc_date: str,
    required_output_language: str,
    required_output_language_label: str,
    theme_plan_markdown: str,
    previous_rounds_markdown: str,
    required_entities: list[str],
    source_overview_packet: str,
) -> list[dict[str, str]]:
    system_contract = (
        "Role: Research Analyst (Overview-First).\n"
        "Mission: Evaluate source overviews and convert them into high-value information units for one core question.\n"
        "Instruction Priority:\n"
        "P1) Schema correctness.\n"
        "P2) Information density and conflict transparency.\n"
        "P3) Language consistency.\n"
        "Hard Constraints:\n"
        "1) Use only SOURCE_OVERVIEW_PACKET.\n"
        "2) Keep analysis scoped to CORE_QUESTION and its information dimensions.\n"
        "3) Distinguish observations from inferences.\n"
        "4) Identify unresolved conflicts and critical information gaps.\n"
        "5) Choose source IDs for full-content arbitration when claims are high-impact, comparative, contradictory, or recency-sensitive.\n"
        "6) Use URL/domain/path/title cues from SOURCE_OVERVIEW_PACKET to estimate source authority and information type.\n"
        "7) For need_content_source_ids, prioritize authoritative URLs first: official documentation, standards/specs, papers/preprints, repositories/model hubs, government/education, and vendor technical docs.\n"
        "8) De-prioritize low-authority commentary or marketing-style pages unless they provide unique, decision-critical information.\n"
        "9) Evaluate temporal relevance for recency-sensitive claims.\n"
        "10) Free-text fields must be strictly in required_output_language.\n"
        "11) Do not mix another language/script except unavoidable proper nouns and product names.\n"
        "12) next_queries must remain strictly focused on CORE_QUESTION and must not introduce new standalone topics.\n"
        "13) required_entities coverage is mandatory when provided: output entity_coverage_complete, covered_entities, and missing_entities.\n"
        "14) Keep required entity strings exact, including version markers (for example qwen3.5, glm4.7).\n"
        "15) If no valid focused query exists, return next_queries as an empty array.\n"
        "16) Return JSON only, exactly matching schema.\n"
        "17) findings must be high-density: each item should include conclusion + condition/boundary + impact.\n"
        "18) next_queries must be de-duplicated and sorted by expected information gain.\n"
        "19) conflict_arbitration items must include topic and status only.\n"
        "20) Do not output citation tokens, evidence-audit sections, or evidence ledgers.\n"
        "Allowed Inputs:\n"
        "- Theme, theme plan, round summaries, overview packet.\n"
        "Failure Policy:\n"
        "- If information is weak or stale for a recency query, lower confidence and propose targeted next queries.\n"
        "Quality Checklist:\n"
        "- Coverage progression, conflict clarity, economical content escalation, calibrated confidence.\n"
        "- Every returned field must add practical value; avoid filler statements."
    ) + _overview_field_contract()
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=_build_style_overlay(
                    stage="overview", style=report_style
                ),
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"THEME:\n{theme}\n\n"
                f"CORE_QUESTION:\n{core_question}\n\n"
                f"REPORT_STYLE_LOCKED:\n{report_style}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"ROUND_INDEX:\n{round_index}\n\n"
                "TIME_CONTEXT:\n"
                f"- current_utc_timestamp={current_utc_timestamp}\n"
                f"- current_utc_date={current_utc_date}\n\n"
                "TEMPORAL_POLICY:\n"
                "- If THEME or prior plans indicate latest/current intent, judge whether each overview is fresh enough.\n"
                "- Treat relative time words (today/this month/this year/recent/latest) against current_utc_date.\n"
                "- For stale evidence under recency intent, request follow-up queries in next_queries.\n"
                "- Any next_queries must directly reduce uncertainty for CORE_QUESTION only.\n\n"
                "SOURCE_SELECTION_POLICY:\n"
                "- Use URL host, path, and URL evidence hint to estimate authority and evidence type.\n"
                "- Prefer authoritative sources for content escalation: docs/specs, papers/preprints, repositories/model hubs, and institutional domains.\n"
                "- Escalate low-authority media/blog pages only when they contain unique evidence not present in authoritative sources.\n\n"
                "DENSITY_POLICY:\n"
                "- findings items must follow: conclusion -> condition/boundary -> impact.\n"
                "- Ban generic filler such as broad restatements with no new information.\n"
                "- next_queries must be non-overlapping and ranked by expected gain.\n\n"
                "LANGUAGE_POLICY:\n"
                f"{_build_language_lock_block(field_name='required_output_language', language_code=required_output_language, language_label=required_output_language_label)}\n"
                "- Keep all free-text fields strictly in required_output_language.\n"
                "- Do not mix another language/script except unavoidable proper nouns.\n\n"
                f"THEME_PLAN_MARKDOWN:\n{theme_plan_markdown}\n\n"
                f"PREVIOUS_ROUNDS_MARKDOWN:\n{previous_rounds_markdown}\n\n"
                "REQUIRED_ENTITIES:\n"
                f"{required_entities}\n\n"
                f"SOURCE_OVERVIEW_PACKET:\n{source_overview_packet}\n\n"
                "Escalation rubric for need_content_source_ids:\n"
                "- Include IDs for sources tied to key conclusions, major conflicts, or model-selection decisions.\n"
                "- Prefer IDs from authoritative URLs (official docs/specs, papers/preprints, repositories/model hubs, government/education domains, vendor technical docs).\n"
                "- Include IDs when overview evidence is vague but potentially important.\n"
                "- For media/blog/secondary pages, include IDs only when they carry unique high-impact facts absent elsewhere.\n"
                "- Include IDs when evidence freshness is uncertain for latest/current requests.\n\n"
                f"{_overview_field_contract()}"
            ),
        },
    ]


def _build_content_messages(
    *,
    theme: str,
    core_question: str,
    report_style: ReportStyle,
    mode_depth_profile: str,
    round_index: int,
    current_utc_timestamp: str,
    current_utc_date: str,
    required_output_language: str,
    required_output_language_label: str,
    theme_plan_markdown: str,
    overview_review_markdown: str,
    required_entities: list[str],
    source_content_packet: str,
) -> list[dict[str, str]]:
    system_contract = (  # noqa: S608
        "Role: Content Arbiter (Full-Content Stage).\n"
        "Mission: Resolve contradictions and raise information completeness for one core question.\n"
        "Instruction Priority:\n"
        "P1) Schema correctness.\n"
        "P2) Content-grounded arbitration quality.\n"
        "P3) Language consistency.\n"
        "Hard Constraints:\n"
        "1) Use only SOURCE_CONTENT_PACKET.\n"
        "2) Keep every judgment aligned to CORE_QUESTION; do not branch into a new standalone topic.\n"
        "3) Mark conflict status conservatively as resolved, unresolved, or insufficient.\n"
        "4) Prefer direct content signals over overview-level assumptions.\n"
        "5) If uncertainty remains, list concrete remaining gaps.\n"
        "6) next_queries must remain strictly focused on CORE_QUESTION and non-redundant.\n"
        "7) For recency-sensitive claims, explicitly account for publication/update-time relevance.\n"
        "8) Free-text fields must be strictly in required_output_language.\n"
        "9) Do not mix another language/script except unavoidable proper nouns and product names.\n"
        "10) resolved_findings must be information-dense: include conclusion + condition/boundary + impact.\n"
        "11) required_entities coverage is mandatory when provided: output entity_coverage_complete, covered_entities, and missing_entities.\n"
        "12) Keep required entity strings exact, including version markers (for example qwen3.5, glm4.7).\n"
        "13) If no valid focused next query exists, return next_queries as an empty array.\n"
        "14) next_queries must be de-duplicated and sorted by expected information gain.\n"
        "15) Return JSON only and match schema exactly.\n"
        "16) conflict_resolutions items must include topic and status only.\n"
        "17) Do not output citation tokens or evidence-audit sections.\n"
        "Allowed Inputs:\n"
        "- Theme, theme plan, overview review, selected content packet.\n"
        "Failure Policy:\n"
        "- If information is insufficient or stale for recency intent, avoid overclaiming and lower confidence.\n"
        "Quality Checklist:\n"
        "- Clear arbitration, traceable rationale, realistic confidence adjustment, gap transparency.\n"
        "- Preserve detail that can later be rendered compactly (fact, conflict, constraint, gap)."
        f"{_content_field_contract()}"
    )
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=_build_style_overlay(stage="content", style=report_style),
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"THEME:\n{theme}\n\n"
                f"CORE_QUESTION:\n{core_question}\n\n"
                f"REPORT_STYLE_LOCKED:\n{report_style}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"ROUND_INDEX:\n{round_index}\n\n"
                "TIME_CONTEXT:\n"
                f"- current_utc_timestamp={current_utc_timestamp}\n"
                f"- current_utc_date={current_utc_date}\n\n"
                "TEMPORAL_POLICY:\n"
                "- Resolve relative time expressions against current_utc_date.\n"
                "- If latest/current intent exists, prefer the most recent trustworthy evidence and flag stale content.\n"
                "- Any next_queries must directly reduce uncertainty for CORE_QUESTION only.\n\n"
                "LANGUAGE_POLICY:\n"
                f"{_build_language_lock_block(field_name='required_output_language', language_code=required_output_language, language_label=required_output_language_label)}\n"
                "- Keep all free-text fields strictly in required_output_language.\n"
                "- Do not mix another language/script except unavoidable proper nouns.\n\n"
                f"THEME_PLAN_MARKDOWN:\n{theme_plan_markdown}\n\n"
                f"OVERVIEW_REVIEW_MARKDOWN:\n{overview_review_markdown}\n\n"
                "REQUIRED_ENTITIES:\n"
                f"{required_entities}\n\n"
                f"SOURCE_CONTENT_PACKET:\n{source_content_packet}\n\n"
                "Arbitration rubric:\n"
                "- resolved: one side is sufficiently better supported by evidence.\n"
                "- unresolved: both sides remain plausible with no decisive tie-break.\n"
                "- insufficient: current evidence cannot adjudicate the claim.\n\n"
                "Output depth rubric:\n"
                "- Prefer specific, decision-useful findings over generic statements.\n"
                "- Every resolved_findings item should include conclusion, condition/boundary, and impact.\n"
                "- Capture trade-offs and boundary conditions when relevant.\n"
                "- Keep confidence_adjustment calibrated to evidence strength.\n\n"
                f"{_content_field_contract()}"
            ),
        },
    ]


def _build_decide_signal_messages(
    *,
    core_question: str,
    mode_depth_profile: str,
    confidence: float,
    coverage_ratio: float,
    unresolved_conflicts: int,
    critical_gaps: int,
    missing_entities: list[str],
    remaining_objectives: list[str],
    search_remaining: int,
    fetch_remaining: int,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Role: stop/continue advisor.\n"
                "Mission: judge whether another round can still generate high-yield information.\n"
                "Rules:\n"
                "1) Make the stop/continue call directly from the full context. Do not imitate a fixed numeric threshold.\n"
                "2) Treat confidence, coverage, and low-gain signals as supporting evidence, not deterministic gates.\n"
                "3) Continue only when expected gain is still meaningful.\n"
                "4) Prefer compact, non-overlapping next queries.\n"
                "5) If stopping, next_queries should usually be empty.\n"
                "6) Omit reason when there is nothing useful to say.\n"
                "7) Output JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CORE_QUESTION:\n{core_question}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                "CURRENT_ROUND:\n"
                f"- confidence={confidence:.3f}\n"
                f"- coverage_ratio={coverage_ratio:.3f}\n"
                f"- unresolved_conflicts={unresolved_conflicts}\n"
                f"- critical_gaps={critical_gaps}\n"
                f"- missing_entities={missing_entities}\n"
                f"- remaining_objectives={remaining_objectives}\n\n"
                "BUDGET_REMAINING:\n"
                f"- search={search_remaining}\n"
                f"- fetch={fetch_remaining}\n"
            ),
        },
    ]


def _build_track_orchestrator_messages(
    *,
    mode_depth_profile: str,
    core_question: str,
    search_remaining: int,
    fetch_remaining: int,
    track_snapshots_markdown: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Role: Multi-track orchestrator.\n"
                "Mission: Assign priority_score and query_width_hint for each track to maximize next-round information gain.\n"
                "Rules:\n"
                "1) Output JSON only.\n"
                "2) Prioritize tracks with high unresolved impact, low confidence, and high decision value.\n"
                "3) De-prioritize tracks showing repeated low gain unless they still block final answer quality.\n"
                "4) query_width_hint must be 1 or 2.\n"
                "5) priority_score must be 0..1.\n"
                "6) Keep reasoning concise and practical."
            ),
        },
        {
            "role": "user",
            "content": (
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"CORE_QUESTION:\n{core_question}\n\n"
                "GLOBAL_BUDGET_REMAINING:\n"
                f"- search={search_remaining}\n"
                f"- fetch={fetch_remaining}\n\n"
                f"TRACK_SNAPSHOTS:\n{track_snapshots_markdown}\n\n"
                "Output format:\n"
                "{\n"
                '  "priorities": [\n'
                '    {"question_id":"q1","priority_score":0.0,"query_width_hint":1,"reason":"..."}\n'
                "  ],\n"
                '  "rationale": "..."\n'
                "}"
            ),
        },
    ]


def _build_gap_closure_messages(
    *,
    core_question: str,
    question_id: str,
    pass_index: int,
    confidence: float,
    coverage_ratio: float,
    unresolved_conflicts: int,
    critical_gaps: int,
    missing_entities: list[str],
    round_notes_markdown: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Role: Gap-closure planner.\n"
                "Mission: Propose compact, high-yield follow-up queries that close the most important unresolved information gaps.\n"
                "Rules:\n"
                "1) Output JSON only.\n"
                "2) Keep queries strictly tied to CORE_QUESTION.\n"
                "3) Prefer non-overlapping queries with clear expected gain.\n"
                "4) If no useful query exists, return an empty array."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CORE_QUESTION:\n{core_question}\n\n"
                f"TRACK_QUESTION_ID:\n{question_id}\n\n"
                f"GAP_CLOSURE_PASS:\n{int(pass_index + 1)}\n\n"
                "LATEST_TRACK_STATUS:\n"
                f"- confidence={confidence}\n"
                f"- coverage_ratio={coverage_ratio}\n"
                f"- unresolved_conflicts={unresolved_conflicts}\n"
                f"- critical_gaps={critical_gaps}\n"
                f"- missing_entities={missing_entities}\n\n"
                f"ROUND_NOTES:\n{round_notes_markdown}\n\n"
                "Output format:\n"
                "{\n"
                '  "queries": ["..."],\n'
                '  "objective": "..."\n'
                "}"
            ),
        },
    ]


def _build_subreport_flow_contract(
    *,
    report_style: ReportStyle,
) -> str:
    style_key = report_style
    if style_key == "decision":
        return (
            "Section flow target:\n"
            "- Verdict Snapshot\n"
            "- Trade-offs\n"
            "- Scenario Recommendation\n"
            "- Risk Triggers\n"
            "- Next Checks"
        )
    if style_key == "execution":
        return (
            "Section flow target:\n"
            "- Goal and Prerequisites\n"
            "- Step Sequence\n"
            "- Validation Criteria\n"
            "- Failure Handling\n"
            "- Next Actions"
        )
    return (
        "Section flow target:\n"
        "- Core Model\n"
        "- Mechanisms\n"
        "- Boundary Cases\n"
        "- Common Misconceptions\n"
        "- Practical Takeaway"
    )


def _build_subreport_messages(
    *,
    target_output_language: str,
    target_output_language_label: str,
    current_utc_date: str,
    core_question: str,
    mode_depth_profile: str,
    report_style: ReportStyle,
    require_insight_card: bool,
    context_packet_markdown: str,
) -> list[dict[str, str]]:
    style_overlay = _build_style_overlay(stage="subreport", style=report_style)
    style_lock_line = f"REPORT_STYLE_LOCKED:\n{report_style}\n\n"
    flow_contract = _build_subreport_flow_contract(
        report_style=report_style,
    )
    system_contract = (
        "Role: Senior research analyst.\n"
        "Mission: Write one external-facing subreport that answers CORE_QUESTION with explicit uncertainty boundaries and high information density.\n"
        "Core principles:\n"
        "1) Stay strictly on CORE_QUESTION.\n"
        "2) Ground every key claim in source-backed information from SUBREPORT_CONTEXT_PACKET_MARKDOWN.\n"
        "3) Distinguish support, conflict, and unknowns.\n"
        "4) Prefer specific facts, dates, conditions, and numbers over generic summary language.\n"
        "5) If information is weak or mixed, narrow the claim and say so directly.\n"
        "Writing quality requirements:\n"
        "1) Start with a direct answer status (confirmed, partial, or unresolved) in plain language.\n"
        "2) Use clear reasoning flow: claim -> support -> implication.\n"
        "3) Explain what disagreements mean for user outcomes.\n"
        "4) Keep prose concise, concrete, and non-repetitive.\n"
        "5) End with targeted next checks that reduce the highest-impact uncertainty.\n"
        "6) Each paragraph must contribute at least one high-value information unit.\n"
        f"{flow_contract}\n"
        "Privacy requirements (strict):\n"
        "1) Output is user-facing; never reveal internal workflow or implementation details.\n"
        "2) Never mention prompt/context packet names, pipeline stages, rounds, query lists, search/fetch calls, stop reasons, confidence/coverage metrics, IDs, or tracking/audit mechanics.\n"
        "3) Never echo internal field names or debug-style labels.\n"
        "4) Do not output sections such as coverage audit or process log.\n"
        "Formatting and time rules:\n"
        "1) Output markdown only.\n"
        "2) Do not use citation tokens or pseudo-citations.\n"
        "3) Use tables only when they materially improve comparison clarity.\n"
        "4) Resolve relative time expressions against CURRENT_UTC_DATE.\n"
        "5) Keep all free text strictly in TARGET_OUTPUT_LANGUAGE.\n"
        "6) Do not mix another language/script except unavoidable proper nouns and product names.\n"
        "7) Return a JSON object with keys: subreport_markdown and track_insight_card.\n"
        "8) subreport_markdown must be valid markdown and must not expose internal process details.\n"
        "9) track_insight_card, when present, must include direct_answer, high_value_points, key_tradeoffs_or_mechanisms, unknowns_and_risks, next_actions.\n"
        "10) high_value_points entries must each contain conclusion, condition, and impact.\n"
        f"11) track_insight_card required={str(bool(require_insight_card)).lower()}.\n"
        f"{_track_insight_card_contract()}"
        "Self-check before final output:\n"
        "- Did each paragraph stay on CORE_QUESTION?\n"
        "- Are uncertainty boundaries explicit and non-overclaiming?\n"
        "- Did I avoid internal-process leakage?"
    )
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=style_overlay,
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_build_language_lock_block(field_name='TARGET_OUTPUT_LANGUAGE', language_code=target_output_language, language_label=target_output_language_label)}\n\n"
                f"CURRENT_UTC_DATE:\n{current_utc_date}\n\n"
                f"CORE_QUESTION:\n{core_question}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"{style_lock_line}"
                "PRIVATE_CONTEXT_NOTICE:\n"
                "- SUBREPORT_CONTEXT_PACKET_MARKDOWN is private working context.\n"
                "- Convert private context into polished user-facing analysis.\n"
                "- Do not expose private metadata, field names, IDs, or process traces.\n\n"
                "OUTPUT_SCHEMA_NOTICE:\n"
                "- Return JSON object with subreport_markdown and track_insight_card.\n"
                "- track_insight_card is required when MODE_DEPTH_PROFILE is research or research-pro.\n"
                "- For each high_value_points item, include conclusion, condition, and impact.\n"
                f"{_track_insight_card_contract()}\n\n"
                f"SUBREPORT_CONTEXT_PACKET_MARKDOWN:\n{context_packet_markdown}"
            ),
        },
    ]


def _build_subreport_update_messages(
    *,
    target_output_language: str,
    target_output_language_label: str,
    current_utc_date: str,
    core_question: str,
    mode_depth_profile: str,
    report_style: ReportStyle,
    require_insight_card: bool,
    update_phase: str,
    current_report_markdown: str,
    context_packet_markdown: str,
) -> list[dict[str, str]]:
    system_contract = (
        "Role: Incremental subreport editor.\n"
        "Mission: Update an existing subreport using only NEW_EVIDENCE_CONTEXT while keeping the report tightly scoped to CORE_QUESTION.\n"
        "Rules:\n"
        "1) Return JSON only.\n"
        "2) action must be one of: update, no_update, stop_after_update.\n"
        "3) stop_after_update is allowed only when CURRENT_REPORT_MARKDOWN is non-empty and the current pass is not the initial draft.\n"
        "4) If action=update, updated_subreport_markdown must contain the full revised report.\n"
        "5) If action=no_update, keep updated_subreport_markdown empty.\n"
        "6) If action=stop_after_update, keep updated_subreport_markdown empty.\n"
        "7) Keep prose strictly in TARGET_OUTPUT_LANGUAGE.\n"
        "8) Use NEW_EVIDENCE_CONTEXT plus CURRENT_REPORT_MARKDOWN only; do not invent new facts.\n"
        "9) Preserve high-value content from CURRENT_REPORT_MARKDOWN unless contradicted or superseded.\n"
        "10) track insight card is required when require_insight_card=true and action=update.\n"
        "11) Choose action=update only when the new evidence materially improves correctness, clarity, or user utility.\n"
        "12) Choose action=no_update when the new evidence is redundant or too weak to justify rewriting.\n"
        "13) Choose action=stop_after_update only after at least one update pass and only when remaining evidence is unlikely to change the answer materially.\n"
        "14) Never return partial markdown; updated_subreport_markdown must be the full revised report when action=update.\n"
        "15) Omit updated_subreport_markdown and updated_track_insight_card when they do not apply to the chosen action.\n"
        f"{_track_insight_card_contract()}"
    )
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=_build_style_overlay(
                    stage="subreport", style=report_style
                ),
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_build_language_lock_block(field_name='TARGET_OUTPUT_LANGUAGE', language_code=target_output_language, language_label=target_output_language_label)}\n\n"
                f"CURRENT_UTC_DATE:\n{current_utc_date}\n\n"
                f"CORE_QUESTION:\n{core_question}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"REPORT_STYLE_LOCKED:\n{report_style}\n\n"
                f"SUBREPORT_UPDATE_PASS:\n{update_phase}\n\n"
                f"TRACK_INSIGHT_CARD_REQUIRED:\n{str(bool(require_insight_card)).lower()}\n\n"
                "ACTION_POLICY:\n"
                "- update: revise the report because the new evidence materially changes quality or correctness.\n"
                "- no_update: keep the current report because the new evidence is redundant, low-confidence, or immaterial.\n"
                "- stop_after_update: end iterative updates because further unseen evidence is unlikely to change the answer materially.\n"
                f"CURRENT_REPORT_MARKDOWN:\n{current_report_markdown or '(empty)'}\n\n"
                f"NEW_EVIDENCE_CONTEXT:\n{context_packet_markdown}"
            ),
        },
    ]


def _build_render_architect_messages(
    *,
    target_output_language: str,
    target_output_language_label: str,
    current_utc_date: str,
    mode_depth_profile: str,
    task_intent: TaskIntent,
    complexity_tier: TaskComplexity,
    report_style: ReportStyle,
    context_packet_markdown: str,
) -> list[dict[str, str]]:
    style_overlay = _build_style_overlay(stage="render_architect", style=report_style)
    style_lock_line = f"REPORT_STYLE_LOCKED:\n{report_style}\n\n"
    scope_lock_contract = research_mode_scope_lock_contract(
        mode_key=mode_depth_profile,
        task_intent=task_intent,
    )
    scope_lock_block = (
        f"\nResearch-mode addendum:\n{scope_lock_contract}"
        if scope_lock_contract
        else ""
    )
    system_contract = (
        "Role: Final-report architect.\n"
        "Mission: produce a JSON-only section blueprint for a polished end-user report.\n"
        "Output requirements:\n"
        "1) Return valid JSON only.\n"
        "2) Return between 5 and 10 sections.\n"
        "3) Ordering is strict: one opening first, body sections in the middle, one closing last.\n"
        "4) section_role must be one of opening/body/closing.\n"
        "5) Every section must include section_id, subhead, section_role, question_ids, scope_requirements, writing_boundaries, must_cover_points, angle, progression_hint.\n"
        "6) Non-core expansion body sections are capped at 1.\n"
        "7) If required question coverage is already complete, do not add expansion sections.\n"
        "8) Under how_to intent, the first two body sections must cover quick-start path and key step sequence.\n"
        "Content-quality requirements:\n"
        "1) Subheads must be concrete, non-overlapping, and non-generic.\n"
        "2) Body sections must form a progressive reasoning flow, not repeated parallel slices.\n"
        "3) must_cover_points must be specific and evidence-seeking.\n"
        "4) writing_boundaries must explicitly block drift and overclaiming.\n"
        "5) If evidence is limited, narrow scope instead of inventing content.\n"
        "Privacy requirements:\n"
        "1) The final report is external-facing; do not expose internal process details.\n"
        "2) Do not design sections about runtime mechanics or internal audits.\n"
        "3) Never include internal metadata in user-facing section intent: question IDs, track IDs, rounds, search/fetch calls, stop reasons, section IDs, or coverage audit.\n"
        "4) Keep language concise and implementation-ready.\n"
        "5) Keep all user-facing blueprint text strictly in TARGET_OUTPUT_LANGUAGE; avoid mixed-script prose except proper nouns."
        f"{scope_lock_block}"
    )
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=style_overlay,
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_build_language_lock_block(field_name='TARGET_OUTPUT_LANGUAGE', language_code=target_output_language, language_label=target_output_language_label)}\n\n"
                f"CURRENT_UTC_DATE:\n{current_utc_date}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"TASK_INTENT:\n{task_intent}\n\n"
                f"COMPLEXITY_TIER:\n{complexity_tier}\n\n"
                f"{style_lock_line}"
                "ARCHITECT_TASK:\n"
                "- Plan only. Do not write report prose.\n"
                "- Optimize for clarity, analytical depth, and decision value.\n"
                "- Keep the blueprint user-facing, not system-facing.\n"
                "- Ensure every question card is covered by at least one body section.\n"
                "- Output schema JSON only.\n\n"
                f"FINAL_CONTEXT_PACKET_MARKDOWN:\n{context_packet_markdown}"
            ),
        },
    ]


def _build_render_writer_messages(
    *,
    target_output_language: str,
    target_output_language_label: str,
    current_utc_date: str,
    mode_depth_profile: str,
    task_intent: TaskIntent,
    complexity_tier: TaskComplexity,
    report_style: ReportStyle,
    section_subhead: str,
    section_prefix_h2: str,
    all_section_plan_markdown: str,
    section_plan_markdown: str,
    context_packet_markdown: str,
) -> list[dict[str, str]]:
    style_overlay = _build_style_overlay(stage="render_writer", style=report_style)
    style_lock_line = f"REPORT_STYLE_LOCKED:\n{report_style}\n\n"
    scope_lock_contract = research_mode_scope_lock_contract(
        mode_key=mode_depth_profile,
        task_intent=task_intent,
    )
    scope_lock_block = (
        f"\nResearch-mode addendum:\n{scope_lock_contract}"
        if scope_lock_contract
        else ""
    )
    system_contract = (
        "Role: Section writer.\n"
        "Mission: write exactly one high-quality report section fragment.\n"
        "Follow CURRENT_SECTION_PLAN_MARKDOWN as a strict contract.\n"
        "Writing goals:\n"
        "1) Be clear, concrete, and task-useful.\n"
        "2) Explain trade-offs and uncertainty boundaries.\n"
        "3) Keep logic explicit: claim -> evidence -> implication.\n"
        "4) Use tables only when they improve comparison or compression of evidence.\n"
        "Formatting rules:\n"
        "1) Output markdown fragment only.\n"
        "2) Use only ### and deeper headings.\n"
        "3) Never output # or ##.\n"
        "4) Never repeat the section H2 title; it is already rendered.\n"
        "5) No citation tokens and no pseudo-citations.\n"
        "6) Keep all prose strictly in TARGET_OUTPUT_LANGUAGE; avoid mixed-script prose except unavoidable proper nouns.\n"
        "Privacy rules (must):\n"
        "1) Never mention internal mechanics, pipeline stages, or prompt/context packet names.\n"
        "2) Never mention internal metadata: track IDs, question IDs, rounds, search calls, fetch calls, stop reasons, coverage audit, or section IDs.\n"
        "3) Write as a polished external report for end users.\n"
        "Quality guardrails:\n"
        "1) Avoid filler, template language, and repetitive phrasing.\n"
        "2) Avoid phrases like 'this report' or 'this section' unless needed for clarity.\n"
        "3) If evidence is insufficient, state limits plainly without exposing internal process.\n"
        "4) Every paragraph must add at least one new high-value information unit.\n"
        "5) The first sentence of each paragraph must state direct task value for the user.\n"
        "6) If a paragraph cannot map to core questions, shrink or remove it instead of expanding."
        f"{scope_lock_block}"
    )
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=style_overlay,
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_build_language_lock_block(field_name='TARGET_OUTPUT_LANGUAGE', language_code=target_output_language, language_label=target_output_language_label)}\n\n"
                f"CURRENT_UTC_DATE:\n{current_utc_date}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"TASK_INTENT:\n{task_intent}\n\n"
                f"COMPLEXITY_TIER:\n{complexity_tier}\n\n"
                "MODE_DEPTH_POLICY:\n"
                "- Keep density high and avoid redundant expansion.\n\n"
                f"{style_lock_line}"
                "SECTION_RENDERING_NOTE:\n"
                "- Final assembler already renders CURRENT_SECTION_PLAN_MARKDOWN.subhead as a `##` title.\n"
                "- Your fragment must not repeat that title.\n\n"
                "PRIVATE_CONTEXT_NOTE:\n"
                "- FINAL_CONTEXT_PACKET_MARKDOWN is private working context.\n"
                "- Do not disclose private metadata in output.\n\n"
                "SECTION_PREFIX_ALREADY_RENDERED:\n"
                f"{section_prefix_h2}\n\n"
                "WRITING_START_RULE:\n"
                "- Continue writing after SECTION_PREFIX_ALREADY_RENDERED.\n"
                "- Do not output SECTION_PREFIX_ALREADY_RENDERED again.\n\n"
                f"CURRENT_SECTION_SUBHEAD_ALREADY_RENDERED_AS_H2:\n{section_subhead}\n\n"
                f"ARCHITECT_REPORT_PLAN_MARKDOWN:\n{all_section_plan_markdown}\n\n"
                f"CURRENT_SECTION_PLAN_MARKDOWN:\n{section_plan_markdown}\n\n"
                f"FINAL_CONTEXT_PACKET_MARKDOWN:\n{context_packet_markdown}"
            ),
        },
    ]


def _build_render_structured_messages(
    *,
    target_output_language: str,
    target_output_language_label: str,
    current_utc_date: str,
    report_style: ReportStyle,
    context_packet_markdown: str,
) -> list[dict[str, str]]:
    style_overlay = _build_style_overlay(stage="render_structured", style=report_style)
    style_lock_line = f"REPORT_STYLE_LOCKED:\n{report_style}\n\n"
    system_contract = (
        "Role: Structured Research Synthesizer.\n"
        "Mission: Build one schema-valid JSON object from FINAL_CONTEXT_PACKET.\n"
        "Rules:\n"
        "1) Output must strictly validate the provided schema.\n"
        "2) Keep all free-text strictly in TARGET_OUTPUT_LANGUAGE.\n"
        "3) Do not mix another language/script except unavoidable proper nouns and product names.\n"
        "4) Keep claims evidence-grounded and uncertainty-aware.\n"
        "5) Resolve relative time terms against CURRENT_UTC_DATE.\n"
        "6) Do not include markdown, code fences, citations, or commentary.\n"
        "7) Do not leak internal process metadata or private context labels.\n"
        "8) Keep free-text fields information-dense and non-repetitive."
    )
    return [
        {
            "role": "system",
            "content": _compose_system_prompt(
                base_contract=system_contract,
                style_overlay=style_overlay,
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_build_language_lock_block(field_name='TARGET_OUTPUT_LANGUAGE', language_code=target_output_language, language_label=target_output_language_label)}\n\n"
                f"CURRENT_UTC_DATE:\n{current_utc_date}\n\n"
                f"{style_lock_line}"
                f"FINAL_CONTEXT_PACKET_MARKDOWN:\n{context_packet_markdown}"
            ),
        },
    ]


def _build_density_gate_messages(
    *,
    target_output_language: str,
    current_utc_date: str,
    mode_depth_profile: str,
    pass_index: int,
    context_packet_markdown: str,
    current_markdown: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Role: Density gate editor.\n"
                "Mission: Rewrite report markdown for higher information density without losing key scope.\n"
                "Rules:\n"
                "1) Keep all core conclusions, uncertainties, and actions.\n"
                "2) Remove filler, repetition, and weak transitions.\n"
                "3) Each paragraph must add non-overlapping value.\n"
                "4) Delete off-topic content before preserving target length.\n"
                "5) Keep prose strictly in TARGET_OUTPUT_LANGUAGE; avoid mixed-script prose except unavoidable proper nouns.\n"
                "6) Do not reveal internal process information.\n"
                "7) Return markdown only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"TARGET_OUTPUT_LANGUAGE:\n{target_output_language}\n\n"
                f"CURRENT_UTC_DATE:\n{current_utc_date}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"DENSITY_PASS_INDEX:\n{int(pass_index + 1)}\n\n"
                "PRIVATE_CONTEXT_NOTICE:\n"
                "- Keep alignment with FINAL_CONTEXT_PACKET_MARKDOWN.\n"
                "- Do not disclose internal metadata.\n\n"
                f"FINAL_CONTEXT_PACKET_MARKDOWN:\n{context_packet_markdown}\n\n"
                f"CURRENT_MARKDOWN:\n{current_markdown}"
            ),
        },
    ]


def build_theme_prompt_messages(
    *,
    ctx: ResearchStepContext,
    now_utc: datetime,
    card_cap: int,
    report_style: ReportStyle,
    engine_selection_context: str = "",
) -> list[dict[str, str]]:
    budget = ctx.run.limits
    mode_depth = ctx.run.limits
    messages = _build_theme_messages(
        theme=ctx.request.themes,
        search_mode=ctx.request.search_mode,
        mode_depth_profile=mode_depth.mode_key,
        current_utc_timestamp=now_utc.isoformat(),
        current_utc_date=now_utc.date().isoformat(),
        max_rounds=budget.max_rounds,
        max_search_calls=budget.max_search_calls,
        max_queries_per_round=budget.max_queries_per_round,
        card_cap=card_cap,
        hinted_style=report_style,
    )
    if engine_selection_context:
        messages[-1]["content"] = messages[-1]["content"].replace(
            "- seed_queries (string[])",
            "- seed_queries ({query, include_sources}[])",
        )
        messages[-1]["content"] += (
            "\n\nENGINE_SELECTION_OUTPUT_RULES:\n"
            "- seed_queries must use objects with query and include_sources.\n"
            "- Prefer include_sources=[] unless a seed query clearly benefits from a targeted evidence route.\n"
            "- Within one question card, prefer seed_queries that explore different source combinations when that improves coverage.\n\n"
            f"{engine_selection_context}"
        )
    return messages


def build_plan_prompt_messages(
    *,
    ctx: ResearchStepContext,
    candidate_queries: list[QuerySourceSpec],
    core_question: str,
    now_utc: datetime,
    last_round_candidates: list[ResearchLinkCandidate],
    engine_selection_context: str = "",
) -> list[dict[str, str]]:
    budget = ctx.run.limits
    mode_depth = ctx.run.limits
    theme_plan = _theme_plan_from_task(ctx.task)
    output_language = theme_plan.output_language
    round_index = ctx.run.round_index
    last_round_candidates_markdown = render_link_candidates_markdown(
        last_round_candidates,
        max_pages=max(1, mode_depth.explore_target_pages_per_round),
        max_links_per_page=6,
    )
    messages = _build_plan_messages(
        theme=ctx.request.themes,
        core_question=core_question,
        report_style=theme_plan.report_style,
        mode_depth_profile=mode_depth.mode_key,
        round_index=round_index,
        current_utc_timestamp=now_utc.isoformat(),
        current_utc_date=now_utc.date().isoformat(),
        required_output_language=output_language,
        required_output_language_label=_resolve_language_label(output_language),
        theme_plan_markdown=render_theme_plan_markdown(theme_plan),
        previous_rounds_markdown=render_rounds_markdown(ctx.run.history, limit=3),
        candidate_queries_markdown=render_queries_markdown(candidate_queries),
        required_entities=list(theme_plan.required_entities),
        search_calls_remaining=max(
            0,
            budget.max_search_calls - ctx.run.search_calls,
        ),
        fetch_calls_remaining=max(
            0,
            budget.max_fetch_calls - ctx.run.fetch_calls,
        ),
        max_queries_this_round=budget.max_queries_per_round,
        allow_explore=_can_attempt_explore(
            ctx=ctx,
            round_index=round_index,
            last_round_candidates=last_round_candidates,
        ),
        explore_target_pages_per_round=mode_depth.explore_target_pages_per_round,
        explore_links_per_page=mode_depth.explore_links_per_page,
        last_round_link_candidates_markdown=last_round_candidates_markdown,
    )
    if engine_selection_context:
        messages[-1]["content"] = (
            messages[-1]["content"]
            .replace(
                '- "search_jobs": [{ "query": "..." }]',
                '- "search_jobs": [{ "query": {"query": "...", "include_sources": []}, "additional_queries": [] }]',
            )
            .replace(
                "- query: one concrete query string, not a prose explanation.",
                "- query: one concrete query object with query and include_sources; query must be a concrete search string, not a prose explanation.",
            )
        )
        messages[-1]["content"] += (
            "\n\nENGINE_SELECTION_OUTPUT_RULES:\n"
            "- query and additional_queries must use objects with query and include_sources.\n"
            "- Keep include_sources=[] as the default unless narrower routing clearly improves evidence quality.\n"
            "- Use routing differences only when they create genuinely different retrieval paths.\n\n"
            f"{engine_selection_context}"
        )
    return messages


def build_link_picker_prompt_messages(
    *,
    core_question: str,
    report_style: ReportStyle,
    mode_depth_profile: str,
    current_utc_date: str,
    source_id: int,
    source_url: str,
    source_title: str,
    max_links_to_select: int,
    candidate_links_markdown: str,
) -> list[dict[str, str]]:
    return _build_link_picker_messages(
        core_question=core_question,
        report_style=report_style,
        mode_depth_profile=mode_depth_profile,
        current_utc_date=current_utc_date,
        source_id=source_id,
        source_url=source_url,
        source_title=source_title,
        max_links_to_select=max_links_to_select,
        candidate_links_markdown=candidate_links_markdown,
    )


def build_overview_prompt_messages(
    *,
    ctx: ResearchStepContext,
    sources: list[ResearchSource],
    now_utc: datetime,
    engine_selection_context: str = "",
) -> list[dict[str, str]]:
    theme_plan = _theme_plan_from_task(ctx.task)
    output_language = theme_plan.output_language
    messages = _build_overview_messages(
        theme=ctx.request.themes,
        core_question=theme_plan.core_question,
        report_style=theme_plan.report_style,
        mode_depth_profile=ctx.run.limits.mode_key,
        round_index=ctx.run.current.round_index if ctx.run.current else 0,
        current_utc_timestamp=now_utc.isoformat(),
        current_utc_date=now_utc.date().isoformat(),
        required_output_language=output_language,
        required_output_language_label=_resolve_language_label(output_language),
        theme_plan_markdown=render_theme_plan_markdown(theme_plan),
        previous_rounds_markdown=render_rounds_markdown(ctx.run.history, limit=3),
        required_entities=list(theme_plan.required_entities),
        source_overview_packet=build_overview_packet(sources=sources),
    )
    if engine_selection_context:
        messages[-1]["content"] += (
            "\n\nENGINE_SELECTION_OUTPUT_RULES:\n"
            "- next_queries must use objects with query and include_sources.\n"
            "- Prefer include_sources=[] unless a follow-up query clearly needs a targeted evidence route.\n"
            "- Only restrict engines when that directly addresses a missing evidence type or freshness gap.\n\n"
            f"{engine_selection_context}"
        )
    return messages


def build_content_prompt_messages(
    *,
    ctx: ResearchStepContext,
    selected_sources: list[ResearchSource],
    source_ids: list[int],
    now_utc: datetime,
    engine_selection_context: str = "",
) -> list[dict[str, str]]:
    overview_review = ctx.run.current.overview_review if ctx.run.current else None
    if overview_review is None:
        raise ValueError("content prompt requires overview review")
    theme_plan = _theme_plan_from_task(ctx.task)
    output_language = theme_plan.output_language
    messages = _build_content_messages(
        theme=ctx.request.themes,
        core_question=theme_plan.core_question,
        report_style=theme_plan.report_style,
        mode_depth_profile=ctx.run.limits.mode_key,
        round_index=ctx.run.current.round_index if ctx.run.current else 0,
        current_utc_timestamp=now_utc.isoformat(),
        current_utc_date=now_utc.date().isoformat(),
        required_output_language=output_language,
        required_output_language_label=_resolve_language_label(output_language),
        theme_plan_markdown=render_theme_plan_markdown(theme_plan),
        overview_review_markdown=render_overview_review_markdown(overview_review),
        required_entities=list(theme_plan.required_entities),
        source_content_packet=build_content_packet(
            sources=selected_sources,
            source_ids=source_ids,
        ),
    )
    if engine_selection_context:
        messages[-1]["content"] += (
            "\n\nENGINE_SELECTION_OUTPUT_RULES:\n"
            "- next_queries must use objects with query and include_sources.\n"
            "- Prefer include_sources=[] unless unresolved conflicts or remaining gaps clearly require targeted routing.\n"
            "- Route only when it strengthens conflict resolution or fills a specific evidence gap.\n\n"
            f"{engine_selection_context}"
        )
    return messages


def build_decide_prompt_messages(
    *,
    ctx: ResearchStepContext,
    engine_selection_context: str = "",
) -> list[dict[str, str]]:
    round_state = ctx.run.current
    if round_state is None:
        return []
    messages = _build_decide_signal_messages(
        core_question=ctx.task.question,
        mode_depth_profile=ctx.run.limits.mode_key,
        confidence=round_state.confidence,
        coverage_ratio=round_state.coverage_ratio,
        unresolved_conflicts=round_state.unresolved_conflicts,
        critical_gaps=round_state.critical_gaps,
        missing_entities=list(round_state.missing_entities),
        remaining_objectives=list(round_state.remaining_objectives),
        search_remaining=max(
            0,
            ctx.run.limits.max_search_calls - ctx.run.search_calls,
        ),
        fetch_remaining=max(
            0,
            ctx.run.limits.max_fetch_calls - ctx.run.fetch_calls,
        ),
    )
    if engine_selection_context:
        messages[-1]["content"] += (
            "\n\nENGINE_SELECTION_OUTPUT_RULES:\n"
            "- next_queries must use objects with query and include_sources.\n"
            "- Prefer include_sources=[] unless the next step clearly needs a specific evidence route.\n"
            "- Keep routing conservative; if uncertainty is high, leave include_sources empty.\n\n"
            f"{engine_selection_context}"
        )
    return messages


def build_track_orchestrator_prompt_messages(
    *,
    root: ResearchStepContext,
    track_map: dict[str, ResearchStepContext],
) -> list[dict[str, str]]:
    budget = root.run.limits
    return _build_track_orchestrator_messages(
        mode_depth_profile=root.run.limits.mode_key,
        core_question=root.task.question,
        search_remaining=max(
            0,
            budget.max_search_calls - root.run.search_calls,
        ),
        fetch_remaining=max(
            0,
            budget.max_fetch_calls - root.run.fetch_calls,
        ),
        track_snapshots_markdown=render_track_snapshot_markdown(track_map),
    )


def build_gap_closure_prompt_messages(
    *,
    card: ResearchQuestionCard,
    track_ctx: ResearchStepContext,
    pass_index: int,
) -> list[dict[str, str]]:
    latest = _latest_round_from_track(track_ctx)
    return _build_gap_closure_messages(
        core_question=track_ctx.task.question or card.question,
        question_id=card.question_id,
        pass_index=pass_index,
        confidence=float(latest.confidence) if latest is not None else 0.0,
        coverage_ratio=float(latest.coverage_ratio) if latest is not None else 0.0,
        unresolved_conflicts=latest.unresolved_conflicts if latest is not None else 0,
        critical_gaps=latest.critical_gaps if latest is not None else 0,
        missing_entities=list(latest.missing_entities) if latest is not None else [],
        round_notes_markdown=render_track_snapshot_markdown(
            {card.question_id: track_ctx}
        ),
    )


def build_subreport_prompt_messages(
    *,
    ctx: ResearchStepContext,
    target_language: str,
    now_utc: datetime,
    source_evidence: list[ResearchSource],
    source_evidence_max_chars: int,
    notes: list[str],
    require_insight_card: bool,
) -> list[dict[str, str]]:
    report_style = ctx.task.style
    theme_plan = _theme_plan_from_task(ctx.task)
    return _build_subreport_messages(
        target_output_language=target_language,
        target_output_language_label=_resolve_language_label(target_language),
        current_utc_date=now_utc.date().isoformat(),
        core_question=ctx.task.question,
        mode_depth_profile=ctx.run.limits.mode_key,
        report_style=report_style,
        require_insight_card=require_insight_card,
        context_packet_markdown=build_subreport_context_packet_markdown(
            theme=ctx.request.themes,
            core_question=ctx.task.question,
            report_style=report_style,
            target_output_language=target_language,
            utc_timestamp=now_utc.isoformat(),
            utc_date=now_utc.date().isoformat(),
            theme_plan=theme_plan,
            rounds=list(ctx.run.history),
            source_evidence=source_evidence,
            source_evidence_max_chars=source_evidence_max_chars,
            notes=notes,
            subreport_objective=_subreport_objective_for_style(
                report_style=report_style
            ),
        ),
    )


def build_subreport_update_prompt_messages(
    *,
    ctx: ResearchStepContext,
    target_language: str,
    now_utc: datetime,
    current_report_markdown: str,
    source_evidence: list[ResearchSource],
    source_evidence_max_chars: int,
    notes: list[str],
    require_insight_card: bool,
    update_phase: str,
) -> list[dict[str, str]]:
    report_style = ctx.task.style
    theme_plan = _theme_plan_from_task(ctx.task)
    return _build_subreport_update_messages(
        target_output_language=target_language,
        target_output_language_label=_resolve_language_label(target_language),
        current_utc_date=now_utc.date().isoformat(),
        core_question=ctx.task.question,
        mode_depth_profile=ctx.run.limits.mode_key,
        report_style=report_style,
        require_insight_card=require_insight_card,
        update_phase=update_phase,
        current_report_markdown=current_report_markdown,
        context_packet_markdown=build_subreport_context_packet_markdown(
            theme=ctx.request.themes,
            core_question=ctx.task.question,
            report_style=report_style,
            target_output_language=target_language,
            utc_timestamp=now_utc.isoformat(),
            utc_date=now_utc.date().isoformat(),
            theme_plan=theme_plan,
            rounds=list(ctx.run.history),
            source_evidence=source_evidence,
            source_evidence_max_chars=source_evidence_max_chars,
            notes=notes,
            subreport_objective=_subreport_objective_for_style(
                report_style=report_style
            ),
        ),
    )


def build_render_architect_prompt_messages(
    *,
    ctx: ResearchStepContext,
    target_language: str,
    now_utc: datetime,
) -> list[dict[str, str]]:
    return _build_render_architect_messages(
        target_output_language=target_language,
        target_output_language_label=_resolve_language_label(target_language),
        current_utc_date=now_utc.date().isoformat(),
        mode_depth_profile=ctx.run.limits.mode_key,
        task_intent=_resolve_task_intent_value(ctx.task.intent),
        complexity_tier=_resolve_task_complexity_value(ctx.task.complexity),
        report_style=ctx.task.style,
        context_packet_markdown=_build_render_context_packet_markdown(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        ),
    )


def build_render_writer_prompt_messages(
    *,
    ctx: ResearchStepContext,
    target_language: str,
    now_utc: datetime,
    architect_output: RenderArchitectOutput,
    section: RenderArchitectSectionPlan,
    track_results: list[ResearchTrackResult],
) -> list[dict[str, str]]:
    section_subhead = section.subhead
    return _build_render_writer_messages(
        target_output_language=target_language,
        target_output_language_label=_resolve_language_label(target_language),
        current_utc_date=now_utc.date().isoformat(),
        mode_depth_profile=ctx.run.limits.mode_key,
        task_intent=_resolve_task_intent_value(ctx.task.intent),
        complexity_tier=_resolve_task_complexity_value(ctx.task.complexity),
        report_style=ctx.task.style,
        section_subhead=section_subhead,
        section_prefix_h2=f"## {section_subhead or 'Section'}",
        all_section_plan_markdown=render_architect_plan_markdown(architect_output),
        section_plan_markdown=render_section_plan_markdown(section),
        context_packet_markdown=_build_render_context_packet_markdown(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
            track_results=track_results,
        ),
    )


def build_render_structured_prompt_messages(
    *,
    ctx: ResearchStepContext,
    target_language: str,
    now_utc: datetime,
) -> list[dict[str, str]]:
    return _build_render_structured_messages(
        target_output_language=target_language,
        target_output_language_label=_resolve_language_label(target_language),
        current_utc_date=now_utc.date().isoformat(),
        report_style=ctx.task.style,
        context_packet_markdown=_build_render_context_packet_markdown(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        ),
    )


def build_density_gate_prompt_messages(
    *,
    ctx: ResearchStepContext,
    markdown: str,
    target_language: str,
    now_utc: datetime,
    pass_index: int,
) -> list[dict[str, str]]:
    return _build_density_gate_messages(
        target_output_language=target_language,
        current_utc_date=now_utc.date().isoformat(),
        mode_depth_profile=ctx.run.limits.mode_key,
        pass_index=pass_index,
        context_packet_markdown=_build_render_context_packet_markdown(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        ),
        current_markdown=markdown,
    )


def _build_render_context_packet_markdown(
    *,
    ctx: ResearchStepContext,
    target_language: str,
    now_utc: datetime,
    track_results: list[ResearchTrackResult] | None = None,
) -> str:
    mode_depth = ctx.run.limits
    theme_plan = _theme_plan_from_task(ctx.task)
    selected_track_results = (
        [item.model_copy(deep=True) for item in track_results]
        if track_results is not None
        else [item.model_copy(deep=True) for item in ctx.result.tracks]
    )
    selected_track_results.sort(
        key=lambda item: (
            float(item.confidence),
            float(item.coverage_ratio),
            -int(item.unresolved_conflicts),
            len(item.key_findings),
        ),
        reverse=True,
    )
    return build_render_final_context_packet_markdown(
        theme=ctx.task.question or ctx.request.themes,
        target_output_language=target_language,
        mode_depth_profile=mode_depth.mode_key,
        utc_timestamp=now_utc.isoformat(),
        utc_date=now_utc.date().isoformat(),
        theme_plan=theme_plan,
        question_cards=[item.model_copy(deep=True) for item in ctx.task.cards],
        track_results=selected_track_results,
        render_objective=_render_objective_for_mode(mode_key=mode_depth.mode_key),
    )


def _normalize_style(raw_style: object | None) -> ReportStyle | None:
    token = str(raw_style).strip().casefold()
    if token not in _STYLE_VALUES:
        return None
    return token  # type: ignore[return-value]


def _latest_round_from_track(
    track_ctx: ResearchStepContext,
) -> ResearchRound | None:
    if track_ctx.run.history:
        return track_ctx.run.history[-1]
    return track_ctx.run.current


def _can_attempt_explore(
    *,
    ctx: ResearchStepContext,
    round_index: int,
    last_round_candidates: list[ResearchLinkCandidate],
) -> bool:
    if round_index <= 1:
        return False
    if ctx.run.limits.max_fetch_calls <= ctx.run.fetch_calls:
        return False
    return len(last_round_candidates) > 0


def _subreport_objective_for_style(*, report_style: ReportStyle) -> str:
    if report_style == "decision":
        return (
            "Produce a decision-focused subreport with scenario-fit recommendations, "
            "trade-offs, and explicit risk triggers."
        )
    if report_style == "execution":
        return (
            "Produce an execution-focused subreport with prerequisites, step sequence, "
            "validation criteria, and failure handling boundaries."
        )
    return (
        "Produce an explainer-focused subreport that clarifies mechanisms, "
        "boundary conditions, and practical understanding."
    )


def _render_objective_for_mode(*, mode_key: str) -> str:
    mode_name = mode_key.casefold()
    if mode_name == "research-fast":
        return (
            "Produce a concise synthesis that answers the theme directly with only "
            "the highest-impact findings."
        )
    if mode_name == "research-pro":
        return (
            "Answer the core user task directly first, then expand to boundary "
            "conditions, tradeoffs, and action-ready implications."
        )
    return (
        "Answer the core user task directly first, then provide a stable "
        "high-density synthesis with clear conclusions, conflicts, uncertainty "
        "boundaries, and actionable implications."
    )


def _resolve_task_intent_value(raw: TaskIntent | str | None) -> TaskIntent:
    token = (raw or "").casefold().replace("-", "_")
    mapping: dict[str, TaskIntent] = {
        "how_to": "how_to",
        "howto": "how_to",
        "comparison": "comparison",
        "compare": "comparison",
        "explainer": "explainer",
        "diagnosis": "diagnosis",
        "other": "other",
    }
    return mapping.get(token, "other")


def _resolve_task_complexity_value(
    raw: TaskComplexity | str | None,
) -> TaskComplexity:
    token = (raw or "").casefold()
    mapping: dict[str, TaskComplexity] = {
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    return mapping.get(token, "medium")


_NONE_BULLET = ["  - (none)"]
_NONE_BULLET_L3 = ["      - (none)"]


def normalize_block_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_scalar_text(value: str) -> str:
    return value


def _render_markdown_bullets(
    values: Iterable[str | QuerySourceSpec], *, indent: str = ""
) -> list[str]:
    out: list[str] = []
    for item in values:
        token = (
            item.query
            + (
                f" [include_sources: {', '.join(item.include_sources)}]"
                if item.include_sources
                else ""
            )
            if isinstance(item, QuerySourceSpec)
            else item
        )
        token = normalize_block_text(token)
        if not token:
            continue
        if "\n" not in token:
            out.append(f"{indent}- {token}")
            continue
        out.append(f"{indent}-")
        out.append(f"{indent}  ```text")
        out.extend(f"{indent}  {line}" for line in token.split("\n"))
        out.append(f"{indent}  ```")
    return out


def render_theme_plan_markdown(
    plan: ResearchThemePlan,
    *,
    include_title: bool = True,
    title_level: int = 3,
    include_question_cards: bool = True,
) -> str:
    lines: list[str] = []
    if include_title:
        level = max(1, title_level)
        lines.append(f"{'#' * level} Theme Plan")
    lines.extend(
        [
            f"- Core question: {_normalize_scalar_text(plan.core_question) or 'n/a'}",
            f"- Report style: {_normalize_scalar_text(plan.report_style) or 'n/a'}",
            f"- Task intent: {_normalize_scalar_text(plan.task_intent) or 'n/a'}",
            f"- Complexity tier: {_normalize_scalar_text(plan.complexity_tier) or 'n/a'}",
            f"- Question card count: {len(plan.question_cards)}",
            f"- Input language: {_normalize_scalar_text(plan.input_language) or 'n/a'}",
            f"- Output language: {_normalize_scalar_text(plan.output_language) or 'n/a'}",
        ]
    )
    lines.append("- Subthemes:")
    lines.extend(_render_markdown_bullets(plan.subthemes, indent="  ") or _NONE_BULLET)
    lines.append("- Required entities:")
    lines.extend(
        _render_markdown_bullets(plan.required_entities, indent="  ") or _NONE_BULLET
    )
    if include_question_cards:
        lines.append("- Question cards:")
        if plan.question_cards:
            for index, card in enumerate(plan.question_cards, start=1):
                lines.extend(
                    _render_theme_plan_card_lines(card=card, index=index, indent="  ")
                )
        else:
            lines.extend(_NONE_BULLET)
    return "\n".join(lines).strip()


def _render_theme_plan_card_lines(
    *,
    card: ResearchThemePlanCard | ResearchQuestionCard,
    index: int,
    indent: str,
) -> list[str]:
    subindent = f"{indent}  "
    leaf_indent = f"{subindent}  "
    lines: list[str] = [
        (
            f"{indent}- Card {index} "
            f"(id={_normalize_scalar_text(getattr(card, 'question_id', '')) or f'q{index}'}, "
            f"priority={getattr(card, 'priority', 0) or 0})"
        ),
        f"{subindent}- Question: {_normalize_scalar_text(getattr(card, 'question', '')) or 'n/a'}",
    ]
    seed_queries = list(getattr(card, "seed_queries", []))
    evidence_focus = list(getattr(card, "evidence_focus", []))
    expected_gain = _normalize_scalar_text(getattr(card, "expected_gain", "")) or "n/a"
    lines.append(f"{subindent}- Seed queries:")
    lines.extend(
        _render_markdown_bullets(seed_queries, indent=leaf_indent)
        or [f"{leaf_indent}- (none)"]
    )
    lines.append(f"{subindent}- Evidence focus:")
    lines.extend(
        _render_markdown_bullets(evidence_focus, indent=leaf_indent)
        or [f"{leaf_indent}- (none)"]
    )
    lines.append(f"{subindent}- Expected gain: {expected_gain}")
    return lines


def render_rounds_markdown(rounds: list[ResearchRound], *, limit: int) -> str:
    selected = rounds[-max(1, limit) :]
    if not selected:
        return "- (none)"
    lines: list[str] = []
    for round_state in selected:
        lines.extend(
            [
                f"### Round {round_state.round_index}",
                (
                    f"- Query strategy: "
                    f"{_normalize_scalar_text(round_state.query_strategy) or 'n/a'}"
                ),
                f"- Result count: {round_state.result_count}",
                f"- Confidence: {float(round_state.confidence):.3f}",
                f"- Coverage ratio: {float(round_state.coverage_ratio):.3f}",
                f"- Entity coverage complete: {round_state.entity_coverage_complete}",
                f"- Unresolved conflicts: {round_state.unresolved_conflicts}",
                f"- Critical gaps: {round_state.critical_gaps}",
                f"- Stop: {round_state.stop}",
                f"- Stop reason: {_normalize_scalar_text(round_state.stop_reason) or 'n/a'}",
                "- Queries:",
            ]
        )
        lines.extend(
            _render_markdown_bullets(round_state.queries, indent="  ") or _NONE_BULLET
        )
        overview_summary = normalize_block_text(round_state.overview_summary)
        content_summary = normalize_block_text(round_state.content_summary)
        lines.append("- Overview summary:")
        if overview_summary:
            lines.extend(
                ["  ```text"]
                + [f"  {line}" for line in overview_summary.split("\n")]
                + ["  ```"]
            )
        else:
            lines.extend(_NONE_BULLET)
        lines.append("- Content summary:")
        if content_summary:
            lines.extend(
                ["  ```text"]
                + [f"  {line}" for line in content_summary.split("\n")]
                + ["  ```"]
            )
        else:
            lines.extend(_NONE_BULLET)
        lines.append("- Missing entities:")
        lines.extend(
            _render_markdown_bullets(round_state.missing_entities, indent="  ")
            or _NONE_BULLET
        )
    return "\n".join(lines).strip()


def render_overview_review_markdown(review: OverviewOutputPayload) -> str:
    lines: list[str] = [
        "### Overview Review",
        f"- Confidence: {float(review.confidence):.3f}",
        f"- Entity coverage complete: {review.entity_coverage_complete}",
        f"- Next query strategy: {_normalize_scalar_text(review.next_query_strategy) or 'n/a'}",
        "- Findings:",
    ]
    lines.extend(_render_markdown_bullets(review.findings, indent="  ") or _NONE_BULLET)
    lines.append("- Covered subthemes:")
    lines.extend(
        _render_markdown_bullets(review.covered_subthemes, indent="  ") or _NONE_BULLET
    )
    lines.append("- Critical gaps:")
    lines.extend(
        _render_markdown_bullets(review.critical_gaps, indent="  ") or _NONE_BULLET
    )
    lines.append("- Need content source IDs:")
    if review.need_content_source_ids:
        lines.extend(f"  - {source_id}" for source_id in review.need_content_source_ids)
    else:
        lines.extend(_NONE_BULLET)
    lines.append("- Conflict arbitration:")
    if review.conflict_arbitration:
        for item in review.conflict_arbitration:
            topic = _normalize_scalar_text(item.topic) or "n/a"
            status = _normalize_scalar_text(item.status) or "n/a"
            lines.append(f"  - topic={topic}; status={status}")
    else:
        lines.extend(_NONE_BULLET)
    lines.append("- Next queries:")
    lines.extend(
        _render_markdown_bullets(review.next_queries, indent="  ") or _NONE_BULLET
    )
    lines.append("- Covered entities:")
    lines.extend(
        _render_markdown_bullets(review.covered_entities, indent="  ") or _NONE_BULLET
    )
    lines.append("- Missing entities:")
    lines.extend(
        _render_markdown_bullets(review.missing_entities, indent="  ") or _NONE_BULLET
    )
    return "\n".join(lines).strip()


def render_queries_markdown(queries: Iterable[str | QuerySourceSpec]) -> str:
    lines = _render_markdown_bullets(queries)
    if not lines:
        return "- (none)"
    return "\n".join(lines).strip()


def render_link_candidates_markdown(
    candidates: list[ResearchLinkCandidate],
    *,
    max_pages: int = 8,
    max_links_per_page: int = 6,
) -> str:
    if not candidates:
        return _NONE_BULLET[0]
    page_limit = max(1, max_pages)
    link_limit = max(1, max_links_per_page)
    lines: list[str] = []
    for item in list(candidates)[:page_limit]:
        main_links = list(item.links or [])
        flat_subpage_links = [
            link
            for group in list(item.subpage_links or [])
            for link in list(group or [])
        ]
        lines.extend(
            [
                f"### Source {item.source_id}",
                f"- URL: {_normalize_scalar_text(item.url) or 'n/a'}",
                f"- Title: {_normalize_scalar_text(item.title) or 'n/a'}",
                f"- Round index: {item.round_index}",
                f"- Main links count: {len(main_links)}",
                f"- Subpage links count: {len(flat_subpage_links)}",
                "- Link samples:",
            ]
        )
        sample_lines: list[str] = [
            (
                "[main] "
                f"{_normalize_scalar_text(link.text) or '(no text)'} -> "
                f"{_normalize_scalar_text(link.url) or 'n/a'}"
            )
            for link in main_links[:link_limit]
        ]
        sample_lines.extend(
            (
                "[subpage] "
                f"{_normalize_scalar_text(link.text) or '(no text)'} -> "
                f"{_normalize_scalar_text(link.url) or 'n/a'}"
            )
            for link in flat_subpage_links[:link_limit]
        )
        lines.extend(
            _render_markdown_bullets(sample_lines, indent="  ") or _NONE_BULLET
        )
    return "\n".join(lines).strip()


def build_overview_packet(*, sources: list[ResearchSource]) -> str:
    blocks: list[str] = []
    for source in sources:
        source_title = source.title
        source_url = source.url
        source_host = _normalize_url_host(source_url)
        source_url_hint = _infer_url_evidence_hint(
            url=source_url,
            title=source_title,
        )
        overview_text = source.overview
        overview_lines = (overview_text or "(none)").split("\n")
        blocks.append(
            "\n".join(
                [
                    f"### Source {source.source_id}",
                    f"- URL: {source_url or 'n/a'}",
                    f"- URL host: {source_host or 'n/a'}",
                    f"- URL evidence hint: {source_url_hint}",
                    f"- Title: {source_title}",
                    f"- Is subpage: {'true' if source.is_subpage else 'false'}",
                    "- Overview:",
                    "  ```text",
                    *[f"  {line}" for line in overview_lines],
                    "  ```",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_content_packet(
    *,
    sources: list[ResearchSource],
    source_ids: list[int],
) -> str:
    wanted = set(source_ids)
    blocks: list[str] = []
    for source in sources:
        if source.source_id not in wanted:
            continue
        content = normalize_block_text(source.content)
        content_lines = (content or "(empty)").split("\n")
        blocks.append(
            "\n".join(
                [
                    f"### Source {source.source_id}",
                    f"- URL: {source.url}",
                    f"- Title: {source.title}",
                    "- Content:",
                    "  ```markdown",
                    *[f"  {line}" for line in content_lines],
                    "  ```",
                ]
            )
        )
    return "\n\n".join(blocks)


def render_track_snapshot_markdown(track_map: dict[str, ResearchStepContext]) -> str:
    lines: list[str] = []
    for question_id, track_ctx in track_map.items():
        latest = (
            track_ctx.run.history[-1]
            if track_ctx.run.history
            else track_ctx.run.current
        )
        lines.extend(
            [
                f"### {question_id}",
                f"- question: {track_ctx.task.question or track_ctx.request.themes}",
                f"- rounds: {len(track_ctx.run.history)}",
                f"- search_calls: {track_ctx.run.search_calls}",
                f"- fetch_calls: {track_ctx.run.fetch_calls}",
                (
                    f"- confidence: {float(latest.confidence):.3f}"
                    if latest is not None
                    else "- confidence: 0.000"
                ),
                (
                    f"- coverage_ratio: {float(latest.coverage_ratio):.3f}"
                    if latest is not None
                    else "- coverage_ratio: 0.000"
                ),
                (
                    f"- unresolved_conflicts: {latest.unresolved_conflicts}"
                    if latest is not None
                    else "- unresolved_conflicts: 0"
                ),
                (
                    f"- critical_gaps: {latest.critical_gaps}"
                    if latest is not None
                    else "- critical_gaps: 0"
                ),
                (
                    f"- stop_reason: {latest.stop_reason or 'n/a'}"
                    if latest is not None
                    else "- stop_reason: n/a"
                ),
            ]
        )
        if latest is not None and latest.remaining_objectives:
            lines.append("- remaining_objectives:")
            lines.extend(
                _render_markdown_bullets(latest.remaining_objectives, indent="  ")
                or _NONE_BULLET
            )
    return "\n".join(lines).strip() or "- (none)"


def build_subreport_context_packet_markdown(
    *,
    theme: str,
    core_question: str,
    report_style: str,
    target_output_language: str,
    utc_timestamp: str,
    utc_date: str,
    theme_plan: ResearchThemePlan,
    rounds: list[ResearchRound],
    source_evidence: list[ResearchSource],
    source_evidence_max_chars: int,
    notes: list[str],
    subreport_objective: str,
) -> str:
    target_output_language_label = _resolve_language_label(target_output_language)
    lines: list[str] = [
        "# Subreport Context Packet",
        "## Theme",
        theme,
        "## Core Question",
        core_question,
        "## Report Style",
        report_style,
        "## Target Output Language",
        f"{target_output_language} ({target_output_language_label})",
        "## Time Context",
        f"- UTC timestamp: {utc_timestamp}",
        f"- UTC date: {utc_date}",
        "## Subreport Objective",
        subreport_objective,
        "## Private Rendering Rules",
        "- SUBREPORT_CONTEXT_PACKET is private working context and must not be exposed verbatim in user-facing output.",
        "- Never expose internal metadata: source IDs, round indexes, query logs, stop reasons, confidence/coverage metrics, or packet labels.",
        "## Theme Plan",
    ]
    lines.append(render_theme_plan_markdown(theme_plan, include_title=False))
    lines.extend(["## Round Trajectory"])
    trajectory = rounds[-8:]
    if trajectory:
        for round_state in trajectory:
            lines.extend(
                [
                    f"### Round {round_state.round_index}",
                    f"- Query strategy: {round_state.query_strategy or 'n/a'}",
                    f"- Result count: {round_state.result_count}",
                    f"- Confidence: {float(round_state.confidence):.3f}",
                    f"- Coverage ratio: {float(round_state.coverage_ratio):.3f}",
                    f"- Unresolved conflicts: {round_state.unresolved_conflicts}",
                    f"- Critical gaps: {round_state.critical_gaps}",
                    f"- Stop: {round_state.stop}",
                    f"- Stop reason: {round_state.stop_reason or 'n/a'}",
                ]
            )
            if round_state.queries:
                lines.append("- Queries:")
                lines.extend(
                    _render_markdown_bullets(round_state.queries[:8], indent="  ")
                    or _NONE_BULLET
                )
            else:
                lines.append("- Queries: (none)")
    else:
        lines.append("- No round trajectory available.")
    lines.extend(["## Source Evidence"])
    source_items = _build_subreport_source_evidence(
        sources=source_evidence,
        max_chars=source_evidence_max_chars,
    )
    if source_items:
        for source in source_items:
            lines.extend(
                [
                    f"### Source {source.source_id}: {source.title or 'Untitled'}",
                    f"- URL: {source.url or 'n/a'}",
                    f"- Round index: {source.round_index}",
                    f"- Is subpage: {source.is_subpage}",
                ]
            )
            lines.append("- Overview:")
            if source.overview:
                lines.extend(
                    ["  ```text"]
                    + [f"  {line}" for line in source.overview.split("\n")]
                    + ["  ```"]
                )
            else:
                lines.append("  - (none)")
            lines.append("- Content excerpt:")
            if source.content:
                lines.extend(
                    [
                        "  ```text",
                        *[f"  {line}" for line in source.content.split("\n")],
                        "  ```",
                    ]
                )
            else:
                lines.append("  - (none)")
    else:
        lines.append("- No source evidence available.")
    lines.extend(["## Notes"])
    if notes:
        lines.extend(f"- {item}" for item in notes)
    else:
        lines.append("- (none)")
    return "\n".join(lines).strip()


def build_render_final_context_packet_markdown(
    *,
    theme: str,
    target_output_language: str,
    mode_depth_profile: str,
    utc_timestamp: str,
    utc_date: str,
    theme_plan: ResearchThemePlan,
    question_cards: list[ResearchQuestionCard],
    track_results: list[ResearchTrackResult],
    render_objective: str,
) -> str:
    target_output_language_label = _resolve_language_label(target_output_language)
    lines: list[str] = [
        "# Final Context Packet",
        "## Theme",
        normalize_block_text(theme) or "n/a",
        "## Target Output Language",
        (
            f"{normalize_block_text(target_output_language) or 'n/a'} "
            f"({target_output_language_label})"
        ),
        "## Mode Depth Profile",
        normalize_block_text(mode_depth_profile) or "n/a",
        "## Time Context",
        f"- UTC timestamp: {utc_timestamp}",
        f"- UTC date: {utc_date}",
        "## Render Objective",
        render_objective,
        "## Theme Plan",
        render_theme_plan_markdown(
            theme_plan,
            include_title=False,
            include_question_cards=False,
        ),
        "## Private Rendering Rules",
        "- Internal metadata is private and must never appear in final user-facing report text.",
        "- Private fields include question IDs, track IDs, rounds, search/fetch call counts, stop reasons, section IDs, and coverage audit status.",
        "## Question Cards",
        render_question_cards_markdown(question_cards),
        "## Track Results",
        _render_track_results_markdown(track_results),
    ]
    return "\n".join(lines).strip()


def render_question_cards_markdown(cards: list[ResearchQuestionCard]) -> str:
    if not cards:
        return _NONE_BULLET[0]
    lines: list[str] = []
    for index, card in enumerate(cards, start=1):
        lines.extend(_render_theme_plan_card_lines(card=card, index=index, indent=""))
    return "\n".join(lines).strip()


def render_architect_plan_markdown(plan: RenderArchitectOutput) -> str:
    lines: list[str] = [
        "### Architect Plan",
        f"- Report objective: {_normalize_scalar_text(plan.report_objective) or 'n/a'}",
        "- Sections:",
    ]
    if not plan.sections:
        lines.extend(_NONE_BULLET)
        return "\n".join(lines).strip()
    for index, section in enumerate(plan.sections, start=1):
        lines.append(
            f"  - Section {index}: "
            f"{_normalize_scalar_text(section.subhead) or _normalize_scalar_text(section.section_id) or 'Section'}"
        )
        lines.extend(
            [
                f"    - section_id: {_normalize_scalar_text(section.section_id) or 'n/a'}",
                f"    - section_role: {_normalize_scalar_text(section.section_role) or 'n/a'}",
                f"    - angle: {_normalize_scalar_text(section.angle) or 'n/a'}",
                (
                    "    - progression_hint: "
                    f"{_normalize_scalar_text(section.progression_hint) or 'n/a'}"
                ),
                "    - question_ids:",
            ]
        )
        lines.extend(
            _render_markdown_bullets(section.question_ids, indent="      ")
            or _NONE_BULLET_L3
        )
        lines.extend(
            [
                "    - scope_requirements:",
            ]
        )
        lines.extend(
            _render_markdown_bullets(section.scope_requirements, indent="      ")
            or _NONE_BULLET_L3
        )
        lines.append("    - writing_boundaries:")
        lines.extend(
            _render_markdown_bullets(section.writing_boundaries, indent="      ")
            or _NONE_BULLET_L3
        )
        lines.append("    - must_cover_points:")
        lines.extend(
            _render_markdown_bullets(section.must_cover_points, indent="      ")
            or _NONE_BULLET_L3
        )
    return "\n".join(lines).strip()


def render_section_plan_markdown(section: RenderArchitectSectionPlan) -> str:
    lines: list[str] = [
        "### Current Section Plan",
        f"- section_id: {_normalize_scalar_text(section.section_id) or 'n/a'}",
        f"- subhead: {_normalize_scalar_text(section.subhead) or 'n/a'}",
        f"- section_role: {_normalize_scalar_text(section.section_role) or 'n/a'}",
        f"- angle: {_normalize_scalar_text(section.angle) or 'n/a'}",
        f"- progression_hint: {_normalize_scalar_text(section.progression_hint) or 'n/a'}",
        "- question_ids:",
    ]
    lines.extend(
        _render_markdown_bullets(section.question_ids, indent="  ") or _NONE_BULLET
    )
    lines.extend(
        [
            "- scope_requirements:",
        ]
    )
    lines.extend(
        _render_markdown_bullets(section.scope_requirements, indent="  ")
        or _NONE_BULLET
    )
    lines.append("- writing_boundaries:")
    lines.extend(
        _render_markdown_bullets(section.writing_boundaries, indent="  ")
        or _NONE_BULLET
    )
    lines.append("- must_cover_points:")
    lines.extend(
        _render_markdown_bullets(section.must_cover_points, indent="  ") or _NONE_BULLET
    )
    return "\n".join(lines).strip()


def _normalize_url_host(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.netloc or ""
    if not host and parsed.path and "://" not in url:
        host = parsed.path.split("/")[0]
    host = host.split("@")[-1].split(":")[0].strip(".").lower()
    return host.removeprefix("www.")


def _infer_url_evidence_hint(*, url: str, title: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = _normalize_url_host(url)
    path = (parsed.path or "").strip().casefold()
    clue_text = f"{host} {path} {title.casefold()}"
    tags: list[str] = []
    if any(
        token in clue_text
        for token in (
            "arxiv.org",
            "doi.org",
            "openreview.net",
            "aclweb.org",
            "ieeexplore",
            "acm.org",
            "springer",
            "nature.com",
            "science.org",
            "jmlr.org",
            "paperswithcode",
        )
    ):
        tags.append("paper_or_research")
    if host.endswith((".edu", ".gov", ".mil")):
        tags.append("institutional_domain")
    if any(
        token in clue_text
        for token in (
            "/docs",
            "/documentation",
            "readthedocs",
            "developer.",
            "/api/",
            "/manual",
            "/reference",
            "/guide",
            "/spec",
            "/standard",
        )
    ):
        tags.append("official_or_technical_docs")
    if any(
        token in clue_text
        for token in ("github.com", "gitlab.com", "bitbucket.org", "huggingface.co")
    ):
        tags.append("repository_or_model_hub")
    if any(
        token in clue_text
        for token in (
            "wikipedia.org",
            "medium.com",
            "substack.com",
            "blog.",
            "/blog/",
            "/news/",
            "/press/",
        )
    ):
        tags.append("general_or_media_content")
    if not tags:
        return "general_web"
    out: list[str] = []
    seen: set[str] = set()
    for item in tags:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return ", ".join(out)


def _build_subreport_source_evidence(
    *,
    sources: list[ResearchSource],
    max_chars: int,
) -> list[ResearchSource]:
    out: list[ResearchSource] = []
    total_limit = max(1, max_chars)
    consumed_chars = 0
    for source in sources:
        content_excerpt = source.content
        if content_excerpt:
            remaining_chars = max(0, total_limit - consumed_chars)
            content_excerpt = content_excerpt[:remaining_chars]
        overview = source.overview
        if overview:
            remaining_chars = max(
                0, total_limit - consumed_chars - len(content_excerpt)
            )
            overview = overview[:remaining_chars]
        projected = consumed_chars + len(content_excerpt) + len(overview)
        if projected > total_limit:
            break
        consumed_chars = projected
        out.append(
            source.model_copy(
                update={
                    "overview": overview,
                    "content": content_excerpt,
                },
                deep=True,
            )
        )
    return out


def _render_track_results_markdown(track_results: list[ResearchTrackResult]) -> str:
    if not track_results:
        return "- (none)"
    lines: list[str] = []
    for index, item in enumerate(track_results, start=1):
        question = normalize_block_text(item.question) or "n/a"
        lines.extend(
            [
                f"### Evidence Cluster {index}",
                f"- Research question: {question}",
                "- Insight card:",
            ]
        )
        insight_card = item.track_insight_card
        if insight_card is None:
            lines.append("  - (none)")
        else:
            lines.append(
                f"  - Direct answer: {normalize_block_text(insight_card.direct_answer) or 'n/a'}"
            )
            lines.append("  - High-value points:")
            if insight_card.high_value_points:
                for point in insight_card.high_value_points:
                    conclusion = normalize_block_text(point.conclusion) or "n/a"
                    condition = normalize_block_text(point.condition) or "n/a"
                    impact = normalize_block_text(point.impact) or "n/a"
                    lines.append(
                        "    - "
                        f"conclusion={conclusion}; condition={condition}; impact={impact}"
                    )
            else:
                lines.append("    - (none)")
            lines.append("  - Tradeoffs/mechanisms:")
            if insight_card.key_tradeoffs_or_mechanisms:
                for token in insight_card.key_tradeoffs_or_mechanisms:
                    text = normalize_block_text(token)
                    if text:
                        lines.append(f"    - {text}")
            else:
                lines.append("    - (none)")
            lines.append("  - Unknowns/risks:")
            if insight_card.unknowns_and_risks:
                for token in insight_card.unknowns_and_risks:
                    text = normalize_block_text(token)
                    if text:
                        lines.append(f"    - {text}")
            else:
                lines.append("    - (none)")
            lines.append("  - Next actions:")
            if insight_card.next_actions:
                for token in insight_card.next_actions:
                    text = normalize_block_text(token)
                    if text:
                        lines.append(f"    - {text}")
            else:
                lines.append("    - (none)")
        lines.append("- Key findings:")
        if item.key_findings:
            for token in item.key_findings:
                finding = normalize_block_text(token)
                if not finding:
                    continue
                if "\n" not in finding:
                    lines.append(f"  - {finding}")
                    continue
                lines.extend(
                    ["  -", "    ```text"]
                    + [f"    {line}" for line in finding.split("\n")]
                    + ["    ```"]
                )
        else:
            lines.append("  - (none)")
        lines.append("- Subreport excerpt:")
        excerpt = normalize_block_text(item.subreport_markdown)
        if excerpt:
            lines.extend(
                ["  ```markdown"]
                + [f"  {line}" for line in excerpt.split("\n")]
                + ["  ```"]
            )
        else:
            lines.append("  - (none)")
    return "\n".join(lines).strip()


__all__ = [
    "build_content_prompt_messages",
    "build_decide_prompt_messages",
    "build_density_gate_prompt_messages",
    "build_gap_closure_prompt_messages",
    "build_link_picker_prompt_messages",
    "build_overview_prompt_messages",
    "build_plan_prompt_messages",
    "build_render_architect_prompt_messages",
    "build_render_structured_prompt_messages",
    "build_render_writer_prompt_messages",
    "build_subreport_prompt_messages",
    "build_subreport_update_prompt_messages",
    "build_theme_prompt_messages",
    "build_track_orchestrator_prompt_messages",
    "normalize_block_text",
]
