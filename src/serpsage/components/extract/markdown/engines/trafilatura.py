from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.extract.markdown.postprocess import (
    finalize_markdown,
    markdown_to_plain,
)
from serpsage.components.extract.markdown.scoring import (
    infer_markdown_stats,
    score_candidate,
)
from serpsage.components.extract.markdown.types import CandidateDoc, StatsMap

if TYPE_CHECKING:
    from serpsage.components.extract.markdown.types import ExtractProfile

try:
    import trafilatura

    _TRAF_AVAILABLE = True
except Exception:  # noqa: BLE001
    trafilatura = None
    _TRAF_AVAILABLE = False


def trafilatura_available() -> bool:
    return bool(_TRAF_AVAILABLE and trafilatura is not None)


def run_trafilatura(
    *,
    html_doc: str,
    profile: ExtractProfile,
    include_secondary_content: bool,
    canonical_secondary_markdown: str,
) -> CandidateDoc | None:
    if not trafilatura_available() or trafilatura is None:
        return None

    try:
        md = trafilatura.extract(
            html_doc,
            output_format="markdown",
            include_tables=True,
            include_links=True,
            favor_precision=True,
            favor_recall=True,
        )
    except Exception as exc:  # noqa: BLE001
        return CandidateDoc(
            markdown="",
            plain_text="",
            extractor_used="trafilatura",
            quality_score=0.0,
            warnings=[f"trafilatura_failed:{type(exc).__name__}"],
            stats={"engine": "trafilatura"},
            primary_chars=0,
            secondary_chars=0,
        )

    primary_markdown = finalize_markdown(markdown=(md or ""), max_chars=profile.max_markdown_chars)
    primary_plain = markdown_to_plain(primary_markdown)

    markdown = primary_markdown
    if include_secondary_content and canonical_secondary_markdown.strip():
        markdown = f"{markdown}\n\n## Secondary Content\n\n{canonical_secondary_markdown}".strip()
    markdown = finalize_markdown(markdown=markdown, max_chars=profile.max_markdown_chars)

    plain = markdown_to_plain(markdown)
    secondary_chars = (
        len(markdown_to_plain(canonical_secondary_markdown)) if include_secondary_content else 0
    )
    inferred = infer_markdown_stats(markdown)
    merged_stats: StatsMap = {
        "engine": "trafilatura",
        "primary_chars": int(len(primary_plain)),
        "secondary_chars": int(secondary_chars),
        **inferred,
    }

    warnings: list[str] = []
    if len(plain) < int(profile.min_plain_chars):
        warnings.append("trafilatura low text output")

    quality = score_candidate(
        markdown=markdown,
        plain_text=plain,
        stats=merged_stats,
        warnings=warnings,
    )

    return CandidateDoc(
        markdown=markdown,
        plain_text=plain,
        extractor_used="trafilatura",
        quality_score=float(quality),
        warnings=warnings,
        stats=merged_stats,
        primary_chars=len(primary_plain),
        secondary_chars=secondary_chars,
    )
