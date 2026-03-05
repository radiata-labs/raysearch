from __future__ import annotations

from typing import Any


def build_theme_schema(*, card_cap: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "detected_input_language",
            "search_language",
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
            "search_language": {"type": "string"},
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
                "maxItems": max(1, card_cap),
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


def build_plan_schema() -> dict[str, Any]:
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
            "query_strategy": {"type": "string"},
            "round_action": {
                "type": "string",
                "enum": ["search", "explore"],
            },
            "explore_target_source_ids": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "integer"},
            },
            "search_jobs": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "intent": {"type": "string"},
                        "mode": {"type": "string"},
                        "include_domains": {
                            "type": "array",
                            "maxItems": 12,
                            "items": {"type": "string"},
                        },
                        "exclude_domains": {
                            "type": "array",
                            "maxItems": 12,
                            "items": {"type": "string"},
                        },
                        "include_text": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string"},
                        },
                        "exclude_text": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string"},
                        },
                        "additional_queries": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


def build_query_language_repair_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["search_jobs"],
        "properties": {
            "search_jobs": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query", "additional_queries"],
                    "properties": {
                        "query": {"type": "string"},
                        "additional_queries": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string"},
                        },
                    },
                },
            }
        },
    }


def build_overview_schema(*, max_queries: int) -> dict[str, Any]:
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
                "items": {"type": "string"},
            },
            "conflict_arbitration": {
                "type": "array",
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["topic", "status"],
                    "properties": {
                        "topic": {"type": "string"},
                        "status": {"type": "string"},
                    },
                },
            },
            "covered_subthemes": {
                "type": "array",
                "maxItems": 16,
                "items": {"type": "string"},
            },
            "entity_coverage_complete": {"type": "boolean"},
            "covered_entities": {
                "type": "array",
                "maxItems": 24,
                "items": {"type": "string"},
            },
            "missing_entities": {
                "type": "array",
                "maxItems": 24,
                "items": {"type": "string"},
            },
            "critical_gaps": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "string"},
            },
            "confidence": {"type": "number"},
            "need_content_source_ids": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "integer", "minimum": 1},
            },
            "next_query_strategy": {"type": "string"},
            "next_queries": {
                "type": "array",
                "maxItems": max(1, max_queries),
                "items": {"type": "string"},
            },
            "stop": {"type": "boolean"},
        },
    }


def build_content_schema(*, max_queries: int) -> dict[str, Any]:
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
                "items": {"type": "string"},
            },
            "conflict_resolutions": {
                "type": "array",
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["status"],
                    "properties": {
                        "status": {"type": "string"},
                    },
                },
            },
            "entity_coverage_complete": {"type": "boolean"},
            "covered_entities": {
                "type": "array",
                "maxItems": 24,
                "items": {"type": "string"},
            },
            "missing_entities": {
                "type": "array",
                "maxItems": 24,
                "items": {"type": "string"},
            },
            "remaining_gaps": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "string"},
            },
            "confidence_adjustment": {"type": "number"},
            "next_query_strategy": {"type": "string"},
            "next_queries": {
                "type": "array",
                "maxItems": max(1, max_queries),
                "items": {"type": "string"},
            },
            "stop": {"type": "boolean"},
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
                "items": {"type": "integer"},
            },
            "reason": {"type": "string"},
        },
    }


__all__ = [
    "build_content_schema",
    "build_link_picker_schema",
    "build_overview_schema",
    "build_plan_schema",
    "build_query_language_repair_schema",
    "build_theme_schema",
]
