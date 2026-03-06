from __future__ import annotations

from serpsage.models.pipeline import ResearchStepContext
from serpsage.utils import clean_whitespace


def resolve_research_model(
    *, ctx: ResearchStepContext, stage: str, fallback: str
) -> str:
    model_settings = ctx.settings.research.models
    stage_to_model = {
        "plan": model_settings.plan,
        "link_select": model_settings.link_select,
        "overview": model_settings.abstract_analyze,
        "abstract": model_settings.abstract_analyze,
        "content": model_settings.content_analyze,
        "synthesize": model_settings.synthesize,
        "markdown": model_settings.markdown,
    }
    token = clean_whitespace(stage_to_model.get(stage, ""))
    return token or fallback


def _append_unique_token(
    *,
    out: list[str],
    seen: set[str],
    raw: object,
    limit: int,
) -> bool:
    value = clean_whitespace(raw if isinstance(raw, str) else str(raw or ""))
    if not value:
        return False
    key = value.casefold()
    if key in seen:
        return False
    seen.add(key)
    out.append(value)
    return len(out) >= max(1, limit)


def normalize_strings(raw: object, *, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if _append_unique_token(out=out, seen=seen, raw=item, limit=limit):
            break
    return out


def merge_strings(*groups: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if _append_unique_token(out=out, seen=seen, raw=item, limit=limit):
                return out
    return out


def normalize_entity_coverage(
    *,
    covered_entities: list[str],
    missing_entities: list[str],
    entity_coverage_complete: bool,
    required_entities: list[str],
) -> tuple[bool, list[str], list[str]]:
    required = normalize_strings(required_entities, limit=24)
    covered = normalize_strings(covered_entities, limit=24)
    if not required:
        return True, covered, []
    required_map = {item.casefold(): item for item in required}
    missing = [
        required_map[item.casefold()]
        for item in normalize_strings(missing_entities, limit=24)
        if item.casefold() in required_map
    ]
    if not missing:
        covered_keys = {item.casefold() for item in covered}
        missing = [item for item in required if item.casefold() not in covered_keys]
    complete = (entity_coverage_complete or not missing_entities) and not missing
    return complete, covered, missing


def normalize_source_ids(raw: object, *, limit: int) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in raw:
        try:
            value = int(item)
        except Exception:  # noqa: S112
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= max(1, limit):
            break
    return out


__all__ = [
    "merge_strings",
    "normalize_entity_coverage",
    "normalize_source_ids",
    "normalize_strings",
    "resolve_research_model",
]
