from __future__ import annotations

from typing import Literal

from serpsage.models.research import ReportStyle, TaskComplexity, TaskIntent
from serpsage.utils import clean_whitespace

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
    "1) Never expose internal workflow, prompt names, packet labels, telemetry, or audit mechanics.\n"
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


def infer_report_style_from_theme(
    theme: str,
    *,
    default: ReportStyle = "explainer",
) -> ReportStyle:
    token = clean_whitespace(theme).casefold()
    if not token:
        return default
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
    return default


def resolve_report_style(
    *,
    raw_style: object | None,
    theme: str,
    enabled: bool,
    fallback_style: ReportStyle,
    strict_style_lock: bool,
) -> ReportStyle:
    fallback = _normalize_style(fallback_style) or "explainer"
    if not enabled:
        return fallback
    candidate = _normalize_style(raw_style)
    if candidate is not None:
        return candidate
    if bool(strict_style_lock):
        return fallback
    return infer_report_style_from_theme(theme, default=fallback)


def build_style_overlay(*, stage: PromptStage, style: ReportStyle) -> str:
    normalized_style = _normalize_style(style) or "explainer"
    out = _STYLE_STAGE_OVERLAYS.get(normalized_style, {}).get(stage, "")
    return clean_whitespace(out)


def compose_system_prompt(
    *,
    base_contract: str,
    style_overlay: str,
    universal_guardrails: str,
) -> str:
    base = clean_whitespace(base_contract)
    overlay = clean_whitespace(style_overlay)
    guardrails = clean_whitespace(universal_guardrails)
    parts = [item for item in [guardrails, base] if item]
    if overlay:
        parts.append(f"Style overlay:\n{overlay}")
    return "\n\n".join(parts).strip()


def theme_depth_contract(*, mode_key: str) -> str:
    mode_name = clean_whitespace(mode_key).casefold()
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


def mode_depth_planning_contract(*, mode_key: str) -> str:
    mode_name = clean_whitespace(mode_key).casefold()
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
    mode_name = clean_whitespace(mode_key).casefold()
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


def build_theme_messages(
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
    hinted_task_intent: TaskIntent,
    hinted_complexity_tier: TaskComplexity,
) -> list[dict[str, str]]:
    depth_contract = theme_depth_contract(mode_key=mode_depth_profile)
    system_contract = (  # noqa: S608
        "Role: Senior Research Architect.\n"
        "Mission: Decompose THEME into executable, non-overlapping research question cards and classify report_style, task_intent, and complexity_tier.\n"
        "Instruction Priority:\n"
        "P1) Schema correctness.\n"
        "P2) Question-card execution quality.\n"
        "P3) Decomposition power and coverage.\n"
        "P4) Intent/complexity classification fit for user task.\n"
        "P5) Report-style fit for user task intent.\n"
        "P6) Language consistency.\n"
        "Step-by-step decomposition method:\n"
        "1) Classify THEME into one primary type: comparison/selection, planning/how-to, diagnosis, trend/forecast, or factual mapping.\n"
        "2) Predict task_intent as one of how_to/comparison/explainer/diagnosis/other.\n"
        "3) Predict complexity_tier as one of low/medium/high based on user task complexity.\n"
        "4) Predict best report_style for user value: decision/explainer/execution.\n"
        "5) Define evidence dimensions before writing cards (for example: performance, cost, reliability, ecosystem, constraints, risk, recency).\n"
        "6) Identify subthemes that ensure high coverage with low overlap.\n"
        "7) Convert subthemes into executable question_cards, each card covering one distinct evidence objective.\n"
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
        "10) task_intent must be exactly one of: how_to, comparison, explainer, diagnosis, other.\n"
        "11) complexity_tier must be exactly one of: low, medium, high.\n"
        "12) Return JSON only.\n"
        "Output Contract (STRICT JSON SHAPE):\n"
        "Top-level keys allowed:\n"
        "- detected_input_language (string)\n"
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
        "Mode-depth contract:\n"
        f"{depth_contract}"
    )
    return [
        {
            "role": "system",
            "content": compose_system_prompt(
                base_contract=system_contract,
                style_overlay=build_style_overlay(stage="theme", style=hinted_style),
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
                "- core_question: one-sentence anchor question.\n"
                "- report_style: classify best user-facing report style as decision/explainer/execution.\n"
                "- task_intent: classify user task intent as how_to/comparison/explainer/diagnosis/other.\n"
                "- complexity_tier: classify task complexity as low/medium/high.\n"
                "- required_entities: exact strings that must be covered by evidence and later summaries.\n"
                "- question_cards: each card is one executable sub-question for one track.\n"
                "- Do not produce top-level seed_queries.\n"
                "- evidence_focus: list what evidence dimensions to prioritize.\n"
                "- expected_gain: concrete learning value from this card.\n\n"
                "Classification Hints:\n"
                f"- hinted_report_style={hinted_style}\n"
                f"- hinted_task_intent={hinted_task_intent}\n"
                f"- hinted_complexity_tier={hinted_complexity_tier}\n\n"
                "Comparison Pattern Example (generic):\n"
                "- Card 1: evaluate candidate A under shared criteria.\n"
                "- Card 2: evaluate candidate B under the same criteria.\n"
                "- Card 3: integrate evidence and decide best fit by scenario."
            ),
        },
    ]


def build_plan_messages(
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
    depth_contract = mode_depth_planning_contract(mode_key=mode_depth_profile)
    system_contract = (  # noqa: S608
        "Role: Principal Research Planner.\n"  # noqa: S608
        "Mission: Select the next-round action and produce focused fallback search jobs.\n"
        "Instruction Priority:\n"
        "P1) Schema correctness and budget adherence.\n"
        "P2) Action fit (search vs explore) for CORE_QUESTION.\n"
        "P3) Information gain per query.\n"
        "P3) Output language consistency.\n"
        "Hard Constraints:\n"
        "1) Return JSON with exactly: query_strategy, round_action, explore_target_source_ids, search_jobs.\n"
        "2) round_action must be exactly search or explore.\n"
        "3) Round 1 must use round_action=search.\n"
        "4) If explore is not allowed in this round, round_action must be search.\n"
        "5) If round_action=explore, select only source IDs from LAST_ROUND_LINK_CANDIDATES and keep list length <= explore_target_pages_per_round.\n"
        "6) If round_action=search, explore_target_source_ids must be an empty array.\n"
        "7) Always return fallback search_jobs even when round_action=explore, unless search budget is already zero.\n"
        "8) All search_jobs must serve CORE_QUESTION. Do not introduce a new independent research question.\n"
        "9) Minimize overlap between jobs while maximizing information gain.\n"
        "10) Respect remaining budget and prioritize unresolved high-value gaps.\n"
        "11) Use deep mode when higher recall or conflict verification is needed.\n"
        "12) Free-text fields must be in the required output language.\n"
        "13) Temporal grounding: interpret relative time words against current UTC date.\n"
        "14) If recency intent exists, include explicit temporal constraints in query text.\n"
        "15) For high-impact claims, prioritize authoritative routes (official documentation, primary sources, standards, vendor announcements, peer-reviewed or institution-backed reports).\n"
        "16) Preserve required_entities exact surface forms and version strings inside query/additional_queries.\n"
        "17) When max_queries_this_round is 1, prefer one deep job with additional_queries for breadth.\n"
        "18) Domain/text route constraints are executable: include_domains, exclude_domains, include_text, exclude_text.\n"
        "19) If focus cannot be preserved, return fewer search_jobs or an empty array; never drift off-topic.\n"
        "20) Return JSON only; no markdown or explanations.\n"
        "Allowed Inputs:\n"
        "- User theme, theme plan, previous round summaries, candidate queries, and last-round link candidates.\n"
        "Failure Policy:\n"
        "- If uncertain about explore value, choose search.\n"
        "- If uncertain, produce fewer but higher-value jobs.\n"
        "- If all candidate queries are off-topic relative to CORE_QUESTION, return search_jobs as an empty array.\n"
        "Quality Checklist:\n"
        "- Action choice is justified by expected information gain.\n"
        "- Distinct intent per job, no near duplicates, conflict-aware targeting.\n"
        "Mode-depth contract:\n"
        f"{depth_contract}"
    )
    return [
        {
            "role": "system",
            "content": compose_system_prompt(
                base_contract=system_contract,
                style_overlay=build_style_overlay(stage="plan", style=report_style),
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
                f"- required_output_language={required_output_language} ({required_output_language_label})\n"
                "- Keep textual fields in the required output language.\n\n"
                "ROUND_ACTION_POLICY:\n"
                "- round 1: must output round_action=search.\n"
                "- round > 1: may choose round_action=explore only when allow_explore=true.\n"
                f"- allow_explore_this_round={str(bool(allow_explore)).lower()}\n"
                f"- explore_target_pages_per_round={int(explore_target_pages_per_round)}\n"
                f"- explore_links_per_page={int(explore_links_per_page)}\n\n"
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
                "- use include/exclude domain and text constraints when that increases authority and precision.\n"
                "- if max_queries_this_round=1, prefer deep mode with additional_queries for one-call breadth.\n\n"
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


def build_link_picker_messages(
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
    system_contract = (
        "Role: Link Selection Analyst.\n"
        "Mission: Select the highest-yield deep-explore links for one source page.\n"
        "Hard Constraints:\n"
        "1) Use both anchor_text and URL when scoring relevance and authority.\n"
        "2) Prioritize official docs/specs, standards, papers/preprints, repositories, and technical references.\n"
        "3) De-prioritize navigation pages, login/signup, pricing, privacy/terms, marketing, and generic index pages.\n"
        "4) Keep selection strictly scoped to CORE_QUESTION.\n"
        "5) Select at most max_links_to_select IDs.\n"
        "6) Return JSON only: selected_link_ids, reason.\n"
        "7) selected_link_ids must reference IDs from CANDIDATE_LINKS only.\n"
        "8) If no candidate is useful, return selected_link_ids as an empty array.\n"
    )
    return [
        {
            "role": "system",
            "content": compose_system_prompt(
                base_contract=system_contract,
                style_overlay=build_style_overlay(stage="plan", style=report_style),
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
                '  "selected_link_ids": [1,2],\n'
                '  "reason": "..."\n'
                "}"
            ),
        },
    ]


def build_overview_messages(
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
        "5) Select source IDs for full-content arbitration when claims are high-impact, comparative, contradictory, or recency-sensitive.\n"
        "6) Use URL/domain/path/title cues from SOURCE_OVERVIEW_PACKET to estimate source authority and information type.\n"
        "7) For need_content_source_ids, prioritize authoritative URLs first: official documentation, standards/specs, papers/preprints, repositories/model hubs, government/education, and vendor technical docs.\n"
        "8) De-prioritize low-authority commentary or marketing-style pages unless they provide unique, decision-critical information.\n"
        "9) Evaluate temporal relevance for recency-sensitive claims.\n"
        "10) Free-text fields must be in the required output language.\n"
        "11) next_queries must remain strictly focused on CORE_QUESTION and must not introduce new standalone topics.\n"
        "12) required_entities coverage is mandatory when provided: output entity_coverage_complete, covered_entities, missing_entities.\n"
        "13) Keep required entity strings exact, including version markers (for example qwen3.5, glm4.7).\n"
        "14) If no valid focused query exists, return next_queries as an empty array.\n"
        "15) Return JSON only, exactly matching schema.\n"
        "16) findings must be high-density: each item should include conclusion + condition/boundary + impact.\n"
        "17) next_queries must be de-duplicated and sorted by expected information gain.\n"
        "18) Do not output citation tokens or evidence-audit sections.\n"
        "Allowed Inputs:\n"
        "- Theme, theme plan, round summaries, overview packet.\n"
        "Failure Policy:\n"
        "- If information is weak or stale for a recency query, lower confidence and propose targeted next queries.\n"
        "Quality Checklist:\n"
        "- Coverage progression, conflict clarity, economical content escalation, calibrated confidence.\n"
        "- Every returned field must add practical value; avoid filler statements."
    )
    return [
        {
            "role": "system",
            "content": compose_system_prompt(
                base_contract=system_contract,
                style_overlay=build_style_overlay(stage="overview", style=report_style),
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
                f"- required_output_language={required_output_language} ({required_output_language_label})\n"
                "- Keep all free-text fields in the required output language.\n\n"
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
                "- Include IDs when evidence freshness is uncertain for latest/current requests."
            ),
        },
    ]


def build_content_messages(
    *,
    theme: str,
    core_question: str,
    report_style: ReportStyle,
    mode_depth_profile: str,
    round_index: str,
    current_utc_timestamp: str,
    current_utc_date: str,
    required_output_language: str,
    required_output_language_label: str,
    theme_plan_markdown: str,
    overview_review_markdown: str,
    required_entities: list[str],
    source_content_packet: str,
) -> list[dict[str, str]]:
    system_contract = (
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
        "8) Free-text fields must be in the required output language.\n"
        "9) resolved_findings must be information-dense: include conclusion + condition/boundary + impact.\n"
        "10) required_entities coverage is mandatory when provided: output entity_coverage_complete, covered_entities, missing_entities.\n"
        "11) Keep required entity strings exact, including version markers (for example qwen3.5, glm4.7).\n"
        "12) If no valid focused next query exists, return next_queries as an empty array.\n"
        "13) next_queries must be de-duplicated and sorted by expected information gain.\n"
        "14) Return JSON only and match schema exactly.\n"
        "15) Do not output citation tokens or evidence-audit sections.\n"
        "Allowed Inputs:\n"
        "- Theme, theme plan, overview review, selected content packet.\n"
        "Failure Policy:\n"
        "- If information is insufficient or stale for recency intent, avoid overclaiming and lower confidence.\n"
        "Quality Checklist:\n"
        "- Clear arbitration, traceable rationale, realistic confidence adjustment, gap transparency.\n"
        "- Preserve detail that can later be rendered compactly (fact, conflict, constraint, gap)."
    )
    return [
        {
            "role": "system",
            "content": compose_system_prompt(
                base_contract=system_contract,
                style_overlay=build_style_overlay(stage="content", style=report_style),
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
                f"- required_output_language={required_output_language} ({required_output_language_label})\n"
                "- Keep all free-text fields in the required output language.\n\n"
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
                "- Keep confidence_adjustment calibrated to evidence strength."
            ),
        },
    ]


def build_decide_signal_messages(
    *,
    core_question: str,
    mode_depth_profile: str,
    confidence: float,
    coverage_ratio: float,
    unresolved_conflicts: int,
    critical_gaps: int,
    missing_entities: list[str],
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
                "1) Continue only when expected gain is meaningful.\n"
                "2) Prefer compact, non-overlapping next queries.\n"
                "3) Output JSON only."
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
                f"- missing_entities={missing_entities}\n\n"
                "BUDGET_REMAINING:\n"
                f"- search={search_remaining}\n"
                f"- fetch={fetch_remaining}\n"
            ),
        },
    ]


def build_track_orchestrator_messages(
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


def build_gap_closure_messages(
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


def build_subreport_flow_contract(
    *,
    report_style: ReportStyle,
    style_applied: bool,
) -> str:
    if not style_applied:
        return (
            "Section flow target:\n"
            "- Direct Answer Status\n"
            "- Evidence Highlights\n"
            "- Conflicts and Uncertainty\n"
            "- Targeted Next Checks"
        )
    style_key = _normalize_style(report_style) or "explainer"
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


def build_subreport_messages(
    *,
    target_output_language: str,
    target_output_language_label: str,
    current_utc_date: str,
    core_question: str,
    mode_depth_profile: str,
    report_style: ReportStyle,
    style_applied: bool,
    require_insight_card: bool,
    context_packet_markdown: str,
) -> list[dict[str, str]]:
    style_overlay = (
        build_style_overlay(stage="subreport", style=report_style)
        if style_applied
        else ""
    )
    style_lock_line = (
        f"REPORT_STYLE_LOCKED:\n{report_style}\n\n" if style_applied else ""
    )
    flow_contract = build_subreport_flow_contract(
        report_style=report_style,
        style_applied=style_applied,
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
        "2) Never mention prompt/context packet names, pipeline stages, rounds, query lists, search/fetch calls, stop reasons, confidence/coverage metrics, IDs, or telemetry/audit mechanics.\n"
        "3) Never echo internal field names or debug-style labels.\n"
        "4) Do not output sections such as coverage audit or process log.\n"
        "Formatting and time rules:\n"
        "1) Output markdown only.\n"
        "2) Do not use citation tokens or pseudo-citations.\n"
        "3) Use tables only when they materially improve comparison clarity.\n"
        "4) Resolve relative time expressions against CURRENT_UTC_DATE.\n"
        "5) Keep all free text in TARGET_OUTPUT_LANGUAGE.\n"
        "6) Return a JSON object with keys: subreport_markdown and track_insight_card.\n"
        "7) subreport_markdown must be valid markdown and must not expose internal process details.\n"
        "8) track_insight_card, when present, must include direct_answer, high_value_points, key_tradeoffs_or_mechanisms, unknowns_and_risks, next_actions.\n"
        "9) high_value_points entries must each contain conclusion, condition, and impact.\n"
        f"10) track_insight_card required={str(bool(require_insight_card)).lower()}.\n"
        "Self-check before final output:\n"
        "- Did each paragraph stay on CORE_QUESTION?\n"
        "- Are uncertainty boundaries explicit and non-overclaiming?\n"
        "- Did I avoid internal-process leakage?"
    )
    return [
        {
            "role": "system",
            "content": compose_system_prompt(
                base_contract=system_contract,
                style_overlay=style_overlay,
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"TARGET_OUTPUT_LANGUAGE_LABEL:\n{target_output_language} ({target_output_language_label})\n\n"
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
                "- For each high_value_points item, include conclusion, condition, and impact.\n\n"
                f"SUBREPORT_CONTEXT_PACKET_MARKDOWN:\n{context_packet_markdown}"
            ),
        },
    ]


def build_render_architect_messages(
    *,
    target_output_language: str,
    target_output_language_label: str,
    current_utc_date: str,
    mode_depth_profile: str,
    task_intent: TaskIntent,
    complexity_tier: TaskComplexity,
    report_style: ReportStyle,
    style_applied: bool,
    section_min: int,
    section_max: int,
    context_packet_markdown: str,
) -> list[dict[str, str]]:
    style_overlay = (
        build_style_overlay(stage="render_architect", style=report_style)
        if style_applied
        else ""
    )
    style_lock_line = (
        f"REPORT_STYLE_LOCKED:\n{report_style}\n\n" if style_applied else ""
    )
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
        f"2) Return {section_min}-{section_max} sections.\n"
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
        "4) Keep language concise and implementation-ready."
        f"{scope_lock_block}"
    )
    return [
        {
            "role": "system",
            "content": compose_system_prompt(
                base_contract=system_contract,
                style_overlay=style_overlay,
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"TARGET_OUTPUT_LANGUAGE_LABEL:\n{target_output_language} ({target_output_language_label})\n\n"
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


def build_render_writer_messages(
    *,
    target_output_language: str,
    target_output_language_label: str,
    current_utc_date: str,
    mode_depth_profile: str,
    task_intent: TaskIntent,
    complexity_tier: TaskComplexity,
    report_style: ReportStyle,
    style_applied: bool,
    section_subhead: str,
    section_prefix_h2: str,
    all_section_plan_markdown: str,
    section_plan_markdown: str,
    context_packet_markdown: str,
) -> list[dict[str, str]]:
    style_overlay = (
        build_style_overlay(stage="render_writer", style=report_style)
        if style_applied
        else ""
    )
    style_lock_line = (
        f"REPORT_STYLE_LOCKED:\n{report_style}\n\n" if style_applied else ""
    )
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
            "content": compose_system_prompt(
                base_contract=system_contract,
                style_overlay=style_overlay,
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"TARGET_OUTPUT_LANGUAGE_LABEL:\n{target_output_language} ({target_output_language_label})\n\n"
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


def build_render_structured_messages(
    *,
    target_output_language: str,
    target_output_language_label: str,
    current_utc_date: str,
    report_style: ReportStyle,
    style_applied: bool,
    context_packet_markdown: str,
) -> list[dict[str, str]]:
    style_overlay = (
        build_style_overlay(stage="render_structured", style=report_style)
        if style_applied
        else ""
    )
    style_lock_line = (
        f"REPORT_STYLE_LOCKED:\n{report_style}\n\n" if style_applied else ""
    )
    system_contract = (
        "Role: Structured Research Synthesizer.\n"
        "Mission: Build one schema-valid JSON object from FINAL_CONTEXT_PACKET.\n"
        "Rules:\n"
        "1) Output must strictly validate the provided schema.\n"
        "2) Keep all free-text in TARGET_OUTPUT_LANGUAGE.\n"
        "3) Keep claims evidence-grounded and uncertainty-aware.\n"
        "4) Resolve relative time terms against CURRENT_UTC_DATE.\n"
        "5) Do not include markdown, code fences, citations, or commentary.\n"
        "6) Do not leak internal process metadata or private context labels.\n"
        "7) Keep free-text fields information-dense and non-repetitive."
    )
    return [
        {
            "role": "system",
            "content": compose_system_prompt(
                base_contract=system_contract,
                style_overlay=style_overlay,
                universal_guardrails=UNIVERSAL_GUARDRAILS,
            ),
        },
        {
            "role": "user",
            "content": (
                f"TARGET_OUTPUT_LANGUAGE_LABEL:\n{target_output_language} ({target_output_language_label})\n\n"
                f"CURRENT_UTC_DATE:\n{current_utc_date}\n\n"
                f"{style_lock_line}"
                f"FINAL_CONTEXT_PACKET_MARKDOWN:\n{context_packet_markdown}"
            ),
        },
    ]


def build_density_gate_messages(
    *,
    target_output_language: str,
    current_utc_date: str,
    mode_depth_profile: str,
    pass_index: int,
    target_chars: int,
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
                "5) Do not reveal internal process information.\n"
                "6) Return markdown only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"TARGET_OUTPUT_LANGUAGE:\n{target_output_language}\n\n"
                f"CURRENT_UTC_DATE:\n{current_utc_date}\n\n"
                f"MODE_DEPTH_PROFILE:\n{mode_depth_profile}\n\n"
                f"DENSITY_PASS_INDEX:\n{int(pass_index + 1)}\n\n"
                f"TARGET_CHARS:\n{int(target_chars)}\n\n"
                "PRIVATE_CONTEXT_NOTICE:\n"
                "- Keep alignment with FINAL_CONTEXT_PACKET_MARKDOWN.\n"
                "- Do not disclose internal metadata.\n\n"
                f"FINAL_CONTEXT_PACKET_MARKDOWN:\n{context_packet_markdown}\n\n"
                f"CURRENT_MARKDOWN:\n{current_markdown}"
            ),
        },
    ]


def _normalize_style(raw_style: object | None) -> ReportStyle | None:
    token = clean_whitespace(str(raw_style or "")).casefold()
    if token not in _STYLE_VALUES:
        return None
    return token  # type: ignore[return-value]


__all__ = [
    "PromptStage",
    "UNIVERSAL_GUARDRAILS",
    "UNIVERSAL_PRIVACY_GUARDRAILS",
    "UNIVERSAL_QUALITY_GUARDRAILS",
    "build_content_messages",
    "build_decide_signal_messages",
    "build_density_gate_messages",
    "build_gap_closure_messages",
    "build_link_picker_messages",
    "build_overview_messages",
    "build_plan_messages",
    "build_render_architect_messages",
    "build_render_structured_messages",
    "build_render_writer_messages",
    "build_subreport_flow_contract",
    "build_subreport_messages",
    "build_style_overlay",
    "build_theme_messages",
    "build_track_orchestrator_messages",
    "compose_system_prompt",
    "infer_report_style_from_theme",
    "mode_depth_planning_contract",
    "resolve_report_style",
    "theme_depth_contract",
]
