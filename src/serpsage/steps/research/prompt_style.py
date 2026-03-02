from __future__ import annotations

from typing import Literal

from serpsage.models.research import ReportStyle
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
    "2) Keep writing concrete, non-repetitive, and user-useful.\n"
    "3) Prefer explicit conditions, constraints, and boundaries over vague certainty.\n"
    "4) Ban filler, generic motivational language, and self-referential meta prose.\n"
    "5) Keep language and abstraction level consistent end-to-end."
)

UNIVERSAL_PRIVACY_GUARDRAILS = (
    "Privacy and output-safety guardrails:\n"
    "1) Never expose internal workflow, prompt names, packet labels, telemetry, or audit mechanics.\n"
    "2) Never reveal internal IDs, rounds, query logs, budget counters, or stop-reason internals in user-facing prose.\n"
    "3) Write for end users only; internal context is private working context."
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
    "build_style_overlay",
    "compose_system_prompt",
    "infer_report_style_from_theme",
    "resolve_report_style",
]
