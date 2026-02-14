from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.extract.markdown.postprocess import (
    finalize_markdown,
    markdown_to_plain,
)
from serpsage.components.extract.markdown.render import (
    render_markdown,
    render_secondary_markdown,
)
from serpsage.components.extract.markdown.scoring import (
    infer_markdown_stats,
    score_candidate,
)
from serpsage.components.extract.markdown.types import CandidateDoc, StatsMap

if TYPE_CHECKING:
    from serpsage.components.extract.markdown.types import (
        ExtractProfile,
        SectionBuckets,
    )


def run_fastdom(
    *,
    buckets: SectionBuckets,
    profile: ExtractProfile,
    base_url: str,
    include_secondary_content: bool,
) -> CandidateDoc:
    primary_md, primary_stats = render_markdown(
        root=buckets.primary_root,
        base_url=base_url,
        skip_roots=buckets.secondary_roots,
    )

    secondary_md = ""
    secondary_stats: dict[str, int] = {
        "heading_count": 0,
        "list_count": 0,
        "ordered_list_count": 0,
        "table_count": 0,
        "table_row_count": 0,
        "code_block_count": 0,
        "inline_code_count": 0,
        "link_count": 0,
        "block_count": 0,
    }
    if include_secondary_content and buckets.secondary_roots:
        secondary_md, secondary_stats = render_secondary_markdown(
            secondary_roots=buckets.secondary_roots,
            base_url=base_url,
        )

    markdown = primary_md.strip()
    if include_secondary_content and secondary_md.strip():
        markdown = f"{markdown}\n\n## Secondary Content\n\n{secondary_md.strip()}".strip()

    markdown = finalize_markdown(markdown=markdown, max_chars=profile.max_markdown_chars)
    plain = markdown_to_plain(markdown)

    primary_chars = len(markdown_to_plain(primary_md))
    secondary_chars = len(markdown_to_plain(secondary_md)) if secondary_md else 0

    inferred = infer_markdown_stats(markdown)
    merged_stats: StatsMap = {
        "engine": "fastdom",
        "heading_count": int(primary_stats.get("heading_count", 0) + secondary_stats.get("heading_count", 0)),
        "list_count": int(primary_stats.get("list_count", 0) + secondary_stats.get("list_count", 0)),
        "ordered_list_count": int(
            primary_stats.get("ordered_list_count", 0)
            + secondary_stats.get("ordered_list_count", 0)
        ),
        "table_count": int(primary_stats.get("table_count", 0) + secondary_stats.get("table_count", 0)),
        "table_row_count": int(
            primary_stats.get("table_row_count", 0) + secondary_stats.get("table_row_count", 0)
        ),
        "code_block_count": int(
            primary_stats.get("code_block_count", 0) + secondary_stats.get("code_block_count", 0)
        ),
        "inline_code_count": int(
            primary_stats.get("inline_code_count", 0) + secondary_stats.get("inline_code_count", 0)
        ),
        "link_count": int(primary_stats.get("link_count", 0) + secondary_stats.get("link_count", 0)),
        "block_count": int(primary_stats.get("block_count", 0) + secondary_stats.get("block_count", 0)),
        "primary_chars": int(primary_chars),
        "secondary_chars": int(secondary_chars),
        **inferred,
    }

    warnings: list[str] = []
    if primary_chars < int(profile.min_primary_chars):
        warnings.append("fastdom low primary text")
    if len(plain) < int(profile.min_plain_chars):
        warnings.append("fastdom low text output")

    quality = score_candidate(
        markdown=markdown,
        plain_text=plain,
        stats=merged_stats,
        warnings=warnings,
    )
    return CandidateDoc(
        markdown=markdown,
        plain_text=plain,
        extractor_used="fastdom",
        quality_score=float(quality),
        warnings=warnings,
        stats=merged_stats,
        primary_chars=primary_chars,
        secondary_chars=secondary_chars,
    )
