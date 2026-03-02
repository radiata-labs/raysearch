from __future__ import annotations

from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.utils import clean_whitespace


def resolve_research_model(
    *, ctx: ResearchStepContext, stage: str, fallback: str
) -> str:
    model_settings = ctx.settings.research.models
    stage_to_model = {
        "plan": model_settings.plan,
        "abstract": model_settings.abstract_analyze,
        "content": model_settings.content_analyze,
        "synthesize": model_settings.synthesize,
        "markdown": model_settings.markdown,
    }
    token = clean_whitespace(stage_to_model.get(stage, ""))
    return token or fallback


def normalize_strings(raw: object, *, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = clean_whitespace(str(item or ""))
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max(1, int(limit)):
            break
    return out


def merge_strings(*groups: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            value = clean_whitespace(str(item or ""))
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= max(1, int(limit)):
                return out
    return out


def build_abstract_packet(
    *, sources: list[ResearchSource], max_abstracts_per_source: int = 5
) -> str:
    blocks: list[str] = []
    for source in sorted(sources, key=lambda item: item.source_id):
        abstracts = [
            text
            for x in source.abstracts[:max_abstracts_per_source]
            if (text := _normalize_block_text(str(x)))
        ]
        abstract_lines: list[str] = []
        if abstracts:
            for index, item in enumerate(abstracts, start=1):
                if "\n" in item:
                    abstract_lines.extend(
                        [
                            f"  - Abstract {index}:",
                            "    ```text",
                            *[f"    {line}" for line in item.split("\n")],
                            "    ```",
                        ]
                    )
                else:
                    abstract_lines.append(f"  - {item}")
        else:
            abstract_lines.append("  - (none)")
        blocks.append(
            "\n".join(
                [
                    f"### Source {int(source.source_id)}",
                    f"- URL: {source.url}",
                    f"- Title: {clean_whitespace(source.title)}",
                    f"- Is subpage: {str(bool(source.is_subpage)).lower()}",
                    "- Abstracts:",
                    "\n".join(abstract_lines),
                ]
            )
        )
    return "\n\n".join(blocks)


def _normalize_block_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_entity_coverage(
    *,
    covered_entities: list[str],
    missing_entities: list[str],
    entity_coverage_complete: bool,
    required_entities: list[str],
) -> tuple[bool, list[str], list[str]]:
    """Normalize and compute entity coverage status.

    Args:
        covered_entities: Entities reported as covered by the LLM.
        missing_entities: Entities reported as missing by the LLM.
        entity_coverage_complete: Coverage complete flag from LLM output.
        required_entities: The canonical list of required entities.

    Returns:
        Tuple of (is_complete, covered_list, missing_list).
    """
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
    complete = bool(
        (bool(entity_coverage_complete) or not missing_entities) and not missing
    )
    return complete, covered, missing
