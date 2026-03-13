from __future__ import annotations

from typing import Any

LANGUAGE_CODE_ENUM = [
    "zh-Hans",
    "zh-Hant",
    "en",
    "ja",
    "ko",
    "fr",
    "de",
    "es",
    "pt",
    "it",
    "ru",
    "ar",
    "hi",
    "tr",
    "other",
]


def _non_empty_string_schema(*, description: str) -> dict[str, Any]:
    return {
        "type": "string",
        "minLength": 1,
        "description": description,
    }


def _query_schema(*, description: str, select_engines: bool) -> dict[str, Any]:
    if not select_engines:
        return _non_empty_string_schema(description=description)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "include_sources"],
        "description": description,
        "properties": {
            "query": _non_empty_string_schema(description="Search query text."),
            "include_sources": {
                "type": "array",
                "description": "Blend engine names to include. Use [] for the default all-open route set.",
                "items": _non_empty_string_schema(
                    description="One engine name to include."
                ),
            },
        },
    }


def build_theme_schema(
    *, card_cap: int, select_engines: bool = False
) -> dict[str, Any]:
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
            "detected_input_language": {
                "type": "string",
                "enum": LANGUAGE_CODE_ENUM,
                "description": "Canonical language code for the user's input language.",
            },
            "core_question": _non_empty_string_schema(
                description="Single-sentence anchor question for the full research task."
            ),
            "report_style": {
                "type": "string",
                "enum": ["decision", "explainer", "execution"],
                "description": "Best report shape for the user task.",
            },
            "task_intent": {
                "type": "string",
                "enum": ["how_to", "comparison", "explainer", "diagnosis", "other"],
                "description": "Primary task intent inferred from the request.",
            },
            "complexity_tier": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Overall research complexity level.",
            },
            "subthemes": {
                "type": "array",
                "maxItems": 12,
                "description": "Distinct coverage dimensions with low overlap.",
                "items": _non_empty_string_schema(
                    description="One subtheme that deserves explicit coverage."
                ),
            },
            "required_entities": {
                "type": "array",
                "maxItems": 16,
                "description": "Exact entity/version surface forms that must remain covered end-to-end.",
                "items": _non_empty_string_schema(
                    description="Exact entity or version string."
                ),
            },
            "question_cards": {
                "type": "array",
                "minItems": 1,
                "maxItems": max(1, card_cap),
                "description": "Executable research cards ordered by importance.",
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
                        "question": _non_empty_string_schema(
                            description="One executable sub-question for a single research track."
                        ),
                        "priority": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "description": "Priority scale: 5=core blocker/highest expected gain, 3=important, 1=nice-to-have.",
                        },
                        "seed_queries": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "description": "High-recall starting queries for this card; avoid near-duplicates.",
                            "items": _query_schema(
                                description="One seed query targeting a distinct evidence route.",
                                select_engines=select_engines,
                            ),
                        },
                        "evidence_focus": {
                            "type": "array",
                            "maxItems": 8,
                            "description": "Concrete evidence dimensions such as cost, benchmark, policy, or recency.",
                            "items": _non_empty_string_schema(
                                description="One evidence dimension or discriminator."
                            ),
                        },
                        "expected_gain": _non_empty_string_schema(
                            description="Specific user value expected from resolving this card."
                        ),
                    },
                },
            },
        },
    }


def build_plan_schema(*, select_engines: bool = False) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "query_strategy",
            "round_action",
            "explore_target_source_ids",
            "search_jobs",
        ],
        "properties": {
            "query_strategy": _non_empty_string_schema(
                description="1-2 sentence rationale describing the next-round search focus and expected gain."
            ),
            "round_action": {
                "type": "string",
                "enum": ["search", "explore"],
                "description": "search=issue new search jobs; explore=follow selected links from prior candidates.",
            },
            "explore_target_source_ids": {
                "type": "array",
                "maxItems": 12,
                "description": "Source IDs to explore this round. Must be empty when round_action=search.",
                "items": {
                    "type": "integer",
                    "minimum": 1,
                },
            },
            "search_jobs": {
                "type": "array",
                "maxItems": 8,
                "description": "Focused search jobs ordered by expected information gain.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query", "intent", "mode", "additional_queries"],
                    "properties": {
                        "query": _query_schema(
                            description="Primary search query for the job.",
                            select_engines=select_engines,
                        ),
                        "intent": {
                            "type": "string",
                            "enum": ["coverage", "deepen", "verify", "refresh"],
                            "description": "coverage=fill missing coverage, deepen=go deeper, verify=resolve contradictions, refresh=check recency.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "deep"],
                            "description": "auto=normal retrieval, deep=broader recall or verification-heavy route.",
                        },
                        "additional_queries": {
                            "type": "array",
                            "maxItems": 8,
                            "description": "Companion queries for adjacent evidence routes inside the same job.",
                            "items": _query_schema(
                                description="One additional focused query.",
                                select_engines=select_engines,
                            ),
                        },
                    },
                },
            },
        },
    }


def build_overview_schema(
    *, max_queries: int, select_engines: bool = False
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "findings",
            "conflict_arbitration",
            "covered_subthemes",
            "entity_coverage_complete",
            "covered_entities",
            "missing_entities",
            "critical_gaps",
            "confidence",
            "need_content_source_ids",
            "next_query_strategy",
            "next_queries",
            "stop",
        ],
        "properties": {
            "findings": {
                "type": "array",
                "maxItems": 20,
                "description": "High-density findings. Each item should state conclusion, condition/boundary, and impact.",
                "items": _non_empty_string_schema(description="One overview finding."),
            },
            "conflict_arbitration": {
                "type": "array",
                "maxItems": 16,
                "description": "Conflict topics already resolved or still unresolved at overview level.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["topic", "status"],
                    "properties": {
                        "topic": _non_empty_string_schema(
                            description="Concrete disputed claim or comparison topic."
                        ),
                        "status": {
                            "type": "string",
                            "enum": ["resolved", "unresolved"],
                            "description": "resolved=overview evidence already breaks the tie; unresolved=needs more work.",
                        },
                    },
                },
            },
            "covered_subthemes": {
                "type": "array",
                "maxItems": 16,
                "description": "Subthemes sufficiently covered by current evidence.",
                "items": _non_empty_string_schema(description="One covered subtheme."),
            },
            "entity_coverage_complete": {
                "type": "boolean",
                "description": "True only when every required entity has meaningful evidence coverage.",
            },
            "covered_entities": {
                "type": "array",
                "maxItems": 24,
                "description": "Required entities already backed by usable evidence.",
                "items": _non_empty_string_schema(
                    description="One covered required entity."
                ),
            },
            "missing_entities": {
                "type": "array",
                "maxItems": 24,
                "description": "Required entities still lacking adequate evidence.",
                "items": _non_empty_string_schema(
                    description="One missing required entity."
                ),
            },
            "critical_gaps": {
                "type": "array",
                "maxItems": 12,
                "description": "Highest-priority unanswered gaps that still block a strong answer.",
                "items": _non_empty_string_schema(
                    description="One critical remaining gap."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Calibrated confidence: 1.0=strongly supported, 0=mixed/unclear, negative=conflicted or weak.",
            },
            "need_content_source_ids": {
                "type": "array",
                "maxItems": 20,
                "description": "Source IDs that should move to full-content review because they are high-impact, conflicting, or uncertain.",
                "items": {"type": "integer", "minimum": 1},
            },
            "next_query_strategy": _non_empty_string_schema(
                description="Short rationale for the next search direction if more work is needed."
            ),
            "next_queries": {
                "type": "array",
                "maxItems": max(1, max_queries),
                "description": "De-duplicated follow-up queries ordered by expected information gain.",
                "items": _query_schema(
                    description="One focused follow-up query.",
                    select_engines=select_engines,
                ),
            },
            "stop": {
                "type": "boolean",
                "description": "True only when the card can safely stop after this overview pass.",
            },
        },
    }


def build_content_schema(
    *, max_queries: int, select_engines: bool = False
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "resolved_findings",
            "conflict_resolutions",
            "entity_coverage_complete",
            "covered_entities",
            "missing_entities",
            "remaining_gaps",
            "confidence_adjustment",
            "next_query_strategy",
            "next_queries",
            "stop",
        ],
        "properties": {
            "resolved_findings": {
                "type": "array",
                "maxItems": 20,
                "description": "Content-grounded findings. Each item should state conclusion, condition/boundary, and impact.",
                "items": _non_empty_string_schema(description="One resolved finding."),
            },
            "conflict_resolutions": {
                "type": "array",
                "maxItems": 16,
                "description": "Arbitration result for each important conflict topic.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["topic", "status"],
                    "properties": {
                        "topic": _non_empty_string_schema(
                            description="Concrete claim or dispute reviewed at content level."
                        ),
                        "status": {
                            "type": "string",
                            "enum": [
                                "resolved",
                                "unresolved",
                                "insufficient",
                                "closed",
                            ],
                            "description": "resolved=conflict settled, unresolved=tie remains, insufficient=not enough evidence, closed=no longer material.",
                        },
                    },
                },
            },
            "entity_coverage_complete": {
                "type": "boolean",
                "description": "True only when every required entity is sufficiently covered after content review.",
            },
            "covered_entities": {
                "type": "array",
                "maxItems": 24,
                "description": "Required entities now supported by adequate evidence.",
                "items": _non_empty_string_schema(
                    description="One covered required entity."
                ),
            },
            "missing_entities": {
                "type": "array",
                "maxItems": 24,
                "description": "Required entities still lacking enough evidence.",
                "items": _non_empty_string_schema(
                    description="One missing required entity."
                ),
            },
            "remaining_gaps": {
                "type": "array",
                "maxItems": 12,
                "description": "Highest-impact gaps that remain after content arbitration.",
                "items": _non_empty_string_schema(description="One remaining gap."),
            },
            "confidence_adjustment": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Signed confidence delta to apply after this content pass.",
            },
            "next_query_strategy": _non_empty_string_schema(
                description="Short rationale for the next research direction if more work is still needed."
            ),
            "next_queries": {
                "type": "array",
                "maxItems": max(1, max_queries),
                "description": "De-duplicated focused follow-up queries ordered by expected information gain.",
                "items": _query_schema(
                    description="One focused follow-up query.",
                    select_engines=select_engines,
                ),
            },
            "stop": {
                "type": "boolean",
                "description": "True only when research can safely stop after content arbitration.",
            },
        },
    }


def build_link_picker_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_link_ids", "reason"],
        "properties": {
            "selected_link_ids": {
                "type": "array",
                "maxItems": 24,
                "description": "Chosen candidate link IDs ordered by usefulness.",
                "items": {"type": "integer"},
            },
            "reason": _non_empty_string_schema(
                description="Brief explanation of why the selected links are highest-yield."
            ),
        },
    }


def build_decide_schema(
    *, max_queries: int, select_engines: bool = False
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "continue_research",
            "high_yield_remaining",
            "next_queries",
            "reason",
        ],
        "properties": {
            "continue_research": {"type": "boolean"},
            "high_yield_remaining": {"type": "boolean"},
            "next_queries": {
                "type": "array",
                "maxItems": max(1, max_queries),
                "items": _query_schema(
                    description="One focused next query.",
                    select_engines=select_engines,
                ),
            },
            "reason": {"type": "string"},
        },
    }


def build_subreport_update_schema(*, require_insight_card: bool) -> dict[str, object]:
    insight_card_schema: dict[str, object] = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "required": [
            "direct_answer",
            "high_value_points",
            "key_tradeoffs_or_mechanisms",
            "unknowns_and_risks",
            "next_actions",
        ],
        "properties": {
            "direct_answer": _non_empty_string_schema(
                description="Best direct answer to the card question in one compact statement."
            ),
            "high_value_points": {
                "type": "array",
                "maxItems": 12,
                "description": "Key takeaways expressed as conclusion, condition, and impact.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["conclusion", "condition", "impact"],
                    "properties": {
                        "conclusion": _non_empty_string_schema(
                            description="Concrete conclusion or directional takeaway."
                        ),
                        "condition": _non_empty_string_schema(
                            description="Scenario, boundary, or condition under which the conclusion holds."
                        ),
                        "impact": _non_empty_string_schema(
                            description="Why the conclusion matters for the user."
                        ),
                    },
                },
            },
            "key_tradeoffs_or_mechanisms": {
                "type": "array",
                "maxItems": 10,
                "description": "Important mechanisms or trade-offs that shape the answer.",
                "items": _non_empty_string_schema(
                    description="One mechanism or trade-off."
                ),
            },
            "unknowns_and_risks": {
                "type": "array",
                "maxItems": 10,
                "description": "Material unknowns, caveats, or failure risks.",
                "items": _non_empty_string_schema(
                    description="One unresolved risk or caveat."
                ),
            },
            "next_actions": {
                "type": "array",
                "maxItems": 10,
                "description": "Concrete next steps that reduce uncertainty or improve execution.",
                "items": _non_empty_string_schema(description="One next action."),
            },
        },
    }
    update_rules: list[dict[str, object]] = [
        {
            "if": {"properties": {"action": {"const": "update"}}},
            "then": {
                "properties": {"updated_subreport_markdown": {"minLength": 1}},
            },
        },
        {
            "if": {
                "properties": {"action": {"enum": ["no_update", "stop_after_update"]}}
            },
            "then": {
                "properties": {
                    "updated_subreport_markdown": {"maxLength": 0},
                },
            },
        },
    ]
    if require_insight_card:
        update_rules.append(
            {
                "if": {"properties": {"action": {"const": "update"}}},
                "then": {
                    "required": ["updated_track_insight_card"],
                    "properties": {
                        "updated_track_insight_card": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "direct_answer",
                                "high_value_points",
                                "key_tradeoffs_or_mechanisms",
                                "unknowns_and_risks",
                                "next_actions",
                            ],
                            "properties": insight_card_schema["properties"],
                        }
                    },
                },
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "updated_subreport_markdown",
            "updated_track_insight_card",
            "summary",
        ],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["update", "no_update", "stop_after_update"],
                "description": "update=apply new evidence, no_update=keep current draft unchanged, stop_after_update=finish after this cycle.",
            },
            "updated_subreport_markdown": {
                "type": "string",
                "description": "Full revised markdown report when action=update; otherwise empty.",
            },
            "updated_track_insight_card": insight_card_schema,
            "summary": _non_empty_string_schema(
                description="Brief explanation of what changed or why no update was needed."
            ),
        },
        "allOf": update_rules,
    }


__all__ = [
    "build_content_schema",
    "build_decide_schema",
    "build_link_picker_schema",
    "build_overview_schema",
    "build_plan_schema",
    "build_subreport_update_schema",
    "build_theme_schema",
]
