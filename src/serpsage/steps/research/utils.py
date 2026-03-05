from __future__ import annotations

from urllib.parse import urlparse

from serpsage.models.pipeline import ResearchSource, ResearchStepContext
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


def build_overview_packet(
    *, sources: list[ResearchSource], max_overview_chars: int = 5000
) -> str:
    blocks: list[str] = []
    for source in sorted(sources, key=lambda item: item.source_id):
        source_title = source.title
        source_url = source.url
        source_host = _normalize_url_host(source_url)
        source_url_hint = _infer_url_evidence_hint(
            url=source_url,
            title=source_title,
        )
        overview_text = source.overview
        if len(overview_text) > max(1, max_overview_chars):
            overview_text = overview_text[: max(1, max_overview_chars)]
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
    complete = (entity_coverage_complete or not missing_entities) and not missing
    return complete, covered, missing
