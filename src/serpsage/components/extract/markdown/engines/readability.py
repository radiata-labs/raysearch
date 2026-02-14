from __future__ import annotations

from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from serpsage.components.extract.markdown.dom import cleanup_dom
from serpsage.components.extract.markdown.postprocess import (
    finalize_markdown,
    markdown_to_plain,
)
from serpsage.components.extract.markdown.render import render_markdown
from serpsage.components.extract.markdown.scoring import (
    infer_markdown_stats,
    score_candidate,
)
from serpsage.components.extract.markdown.types import CandidateDoc, StatsMap
from serpsage.text.normalize import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.extract.markdown.types import ExtractProfile

try:
    from readability import Document as ReadabilityDocument

    _READABILITY_AVAILABLE = True
except Exception:  # noqa: BLE001
    ReadabilityDocument = None
    _READABILITY_AVAILABLE = False


def readability_available() -> bool:
    return bool(_READABILITY_AVAILABLE and ReadabilityDocument is not None)


def run_readability(
    *,
    html_doc: str,
    base_url: str,
    profile: ExtractProfile,
    include_secondary_content: bool,
    canonical_secondary_markdown: str,
) -> CandidateDoc | None:
    if not readability_available() or ReadabilityDocument is None:
        return None

    try:
        doc = ReadabilityDocument(html_doc)
        article_html = doc.summary() or ""
        title = clean_whitespace(doc.short_title() or "")
    except Exception as exc:  # noqa: BLE001
        return CandidateDoc(
            markdown="",
            plain_text="",
            extractor_used="readability",
            quality_score=0.0,
            warnings=[f"readability_failed:{type(exc).__name__}"],
            stats={"engine": "readability"},
            primary_chars=0,
            secondary_chars=0,
        )

    soup = BeautifulSoup(article_html, "html.parser")
    cleanup_dom(soup)
    markdown, stats = render_markdown(root=soup, base_url=base_url)
    if title:
        markdown = f"# {title}\n\n{markdown}".strip()
        stats["heading_count"] = int(stats.get("heading_count", 0)) + 1

    primary_markdown = finalize_markdown(markdown=markdown, max_chars=profile.max_markdown_chars)
    primary_plain = markdown_to_plain(primary_markdown)

    output_markdown = primary_markdown
    if include_secondary_content and canonical_secondary_markdown.strip():
        output_markdown = f"{output_markdown}\n\n## Secondary Content\n\n{canonical_secondary_markdown}".strip()
    output_markdown = finalize_markdown(
        markdown=output_markdown,
        max_chars=profile.max_markdown_chars,
    )

    plain = markdown_to_plain(output_markdown)
    secondary_chars = (
        len(markdown_to_plain(canonical_secondary_markdown)) if include_secondary_content else 0
    )
    inferred = infer_markdown_stats(output_markdown)
    merged_stats: StatsMap = {
        "engine": "readability",
        "heading_count": int(stats.get("heading_count", 0)),
        "table_count": int(stats.get("table_count", 0)),
        "table_row_count": int(stats.get("table_row_count", 0)),
        "code_block_count": int(stats.get("code_block_count", 0)),
        "inline_code_count": int(stats.get("inline_code_count", 0)),
        "link_count": int(stats.get("link_count", 0)),
        "list_count": int(stats.get("list_count", 0)),
        "ordered_list_count": int(stats.get("ordered_list_count", 0)),
        "primary_chars": int(len(primary_plain)),
        "secondary_chars": int(secondary_chars),
        **inferred,
    }

    warnings: list[str] = []
    if len(plain) < int(profile.min_plain_chars):
        warnings.append("readability low text output")

    quality = score_candidate(
        markdown=output_markdown,
        plain_text=plain,
        stats=merged_stats,
        warnings=warnings,
    )

    return CandidateDoc(
        markdown=output_markdown,
        plain_text=plain,
        extractor_used="readability",
        quality_score=float(quality),
        warnings=warnings,
        stats=merged_stats,
        primary_chars=len(primary_plain),
        secondary_chars=secondary_chars,
    )
