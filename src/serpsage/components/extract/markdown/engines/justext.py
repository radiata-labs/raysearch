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
    import justext

    _JUSTEXT_AVAILABLE = True
except Exception:  # noqa: BLE001
    justext = None
    _JUSTEXT_AVAILABLE = False


def justext_available() -> bool:
    return bool(_JUSTEXT_AVAILABLE and justext is not None)


def run_justext(
    *,
    html_doc: str,
    profile: ExtractProfile,
    include_secondary_content: bool,
    canonical_secondary_markdown: str,
) -> CandidateDoc | None:
    if not justext_available() or justext is None:
        return None

    try:
        stoplist = justext.get_stoplist("English")
        paragraphs = justext.justext(html_doc.encode("utf-8", errors="ignore"), stoplist)
        blocks = [
            str(paragraph.text).strip()
            for paragraph in paragraphs
            if not bool(getattr(paragraph, "is_boilerplate", False))
        ]
    except Exception as exc:  # noqa: BLE001
        return CandidateDoc(
            markdown="",
            plain_text="",
            extractor_used="justext",
            quality_score=0.0,
            warnings=[f"justext_failed:{type(exc).__name__}"],
            stats={"engine": "justext"},
            primary_chars=0,
            secondary_chars=0,
        )

    primary_markdown = finalize_markdown(
        markdown="\n\n".join(block for block in blocks if block),
        max_chars=profile.max_markdown_chars,
    )
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
        "engine": "justext",
        "primary_chars": int(len(primary_plain)),
        "secondary_chars": int(secondary_chars),
        **inferred,
    }

    warnings: list[str] = []
    if len(plain) < int(profile.min_plain_chars):
        warnings.append("justext low text output")

    quality = score_candidate(
        markdown=markdown,
        plain_text=plain,
        stats=merged_stats,
        warnings=warnings,
    )

    return CandidateDoc(
        markdown=markdown,
        plain_text=plain,
        extractor_used="justext",
        quality_score=float(quality),
        warnings=warnings,
        stats=merged_stats,
        primary_chars=len(primary_plain),
        secondary_chars=secondary_chars,
    )
