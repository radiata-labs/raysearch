from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast
from typing_extensions import override

from serpsage.components.extract.base import ExtractorBase
from serpsage.components.extract.html.dom import (
    HtmlSnapshot,
    build_html_snapshot,
    collect_image_links_inventory,
    collect_links_inventory,
)
from serpsage.components.extract.html.postprocess import (
    finalize_markdown,
    markdown_to_abstract_text,
    markdown_to_text,
)
from serpsage.components.extract.html.render import (
    RenderStats,
    TrafilaturaMetadata,
    empty_render_stats,
    extract_trafilatura_markdown,
    extract_trafilatura_metadata,
    merge_render_stats,
    render_fragment_markdown,
)
from serpsage.components.extract.utils import (
    decode_best_effort,
    guess_apparent_encoding,
)
from serpsage.components.fetch.utils import classify_content_kind
from serpsage.core.runtime import Runtime
from serpsage.models.extract import (
    ExtractContentOptions,
    ExtractContentTag,
    ExtractedDocument,
)
from serpsage.settings.models import AppSettings
from serpsage.utils import clean_whitespace

StatValue: TypeAlias = int | float | str | bool
StatsMap: TypeAlias = dict[str, StatValue]

_SEMANTIC_ORDER: tuple[ExtractContentTag, ...] = (
    "metadata",
    "header",
    "navigation",
    "banner",
    "body",
    "sidebar",
    "footer",
)
_MIN_HTML_CAPTURE_CHARS = 5_000_000


@dataclass(slots=True)
class ExtractProfile:
    max_markdown_chars: int
    max_html_chars: int
    min_text_chars: int
    min_primary_chars: int
    min_total_chars_with_secondary: int
    link_max_count: int
    link_keep_hash: bool


@dataclass(slots=True)
class RenderBundle:
    markdown: str
    render_stats: RenderStats
    secondary_markdown: str


class HtmlExtractor(ExtractorBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        self._profile = build_extract_profile(settings=self.settings)

    @override
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        content_options: ExtractContentOptions | None = None,
        include_secondary_content: bool = False,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        profile = self._profile
        kind = classify_content_kind(
            content_type=content_type,
            url=url,
            content=content,
        )
        if kind == "pdf":
            raise ValueError(
                "MarkdownExtractor does not handle PDF; use AutoExtractor/PdfExtractor"
            )
        apparent = guess_apparent_encoding(content)
        decoded, decoded_kind = decode_best_effort(
            content,
            content_type=content_type,
            apparent_encoding=apparent,
        )
        if kind == "unknown":
            kind = "html" if decoded_kind == "html" else "text"
        options = content_options or ExtractContentOptions(
            detail="full" if include_secondary_content else "concise"
        )
        if kind == "text":
            return self._attach_abstract_markdown(self._extract_text(text=decoded))
        if kind != "html":
            return self._attach_abstract_markdown(
                ExtractedDocument(
                    content_kind="binary",
                    extractor_used="binary",
                    warnings=["unsupported binary content"],
                    stats={"primary_chars": 0, "secondary_chars": 0},
                )
            )
        raw_html = decoded[: int(profile.max_html_chars)]
        snapshot = build_html_snapshot(raw_html=raw_html, base_url=url)
        metadata = extract_trafilatura_metadata(raw_html=raw_html, url=url)
        body_markdown = extract_trafilatura_markdown(
            raw_html=raw_html,
            url=url,
            max_chars=profile.max_markdown_chars,
        )
        selected_tags = self._resolve_selected_tags(options=options)
        rendered = self._render_document(
            snapshot=snapshot,
            metadata=metadata,
            body_markdown=body_markdown,
            options=options,
            selected_tags=selected_tags,
            base_url=url,
        )
        markdown = finalize_markdown(
            markdown=rendered.markdown,
            max_chars=profile.max_markdown_chars,
        )
        markdown = self._prepend_title_heading(
            markdown=markdown,
            title=metadata.title,
            include_html_tags=bool(options.include_html_tags),
            max_chars=profile.max_markdown_chars,
        )
        self._assert_filtered_output_nonempty(
            markdown=markdown,
            min_text_chars=profile.min_text_chars,
        )
        include_secondary = "sidebar" in selected_tags
        stats = self._build_stats(
            markdown=markdown,
            body_markdown=body_markdown,
            secondary_markdown=rendered.secondary_markdown,
            render_stats=rendered.render_stats,
            detail=options.detail,
            include_secondary=include_secondary,
            selected_tags=selected_tags,
        )
        warnings = self._build_warnings(
            primary_chars=int(stats["primary_chars"]),
            text_chars=int(stats["text_chars"]),
            include_secondary=include_secondary,
        )
        return self._attach_abstract_markdown(
            ExtractedDocument(
                markdown=markdown,
                title=metadata.title,
                published_date=metadata.published_date,
                author=metadata.author,
                image=metadata.image,
                favicon=snapshot.favicon,
                content_kind="html",
                extractor_used=str(stats["engine_chain"]),
                warnings=warnings,
                stats=stats,
                links=(
                    collect_links_inventory(
                        snapshot=snapshot,
                        base_url=url,
                        include_secondary_content=include_secondary,
                        max_links=profile.link_max_count,
                        keep_hash=profile.link_keep_hash,
                    )
                    if collect_links
                    else []
                ),
                image_links=(
                    collect_image_links_inventory(
                        snapshot=snapshot,
                        base_url=url,
                        include_secondary_content=include_secondary,
                        max_links=profile.link_max_count,
                        keep_hash=profile.link_keep_hash,
                    )
                    if collect_images
                    else []
                ),
            )
        )

    def _extract_text(self, *, text: str) -> ExtractedDocument:
        lines = [
            cleaned for line in text.splitlines() if (cleaned := clean_whitespace(line))
        ]
        markdown = finalize_markdown(
            markdown="\n\n".join(lines),
            max_chars=int(self._profile.max_markdown_chars),
        )
        text_value = markdown_to_text(markdown)
        return ExtractedDocument(
            markdown=markdown,
            content_kind="text",
            extractor_used="text",
            stats={
                "primary_chars": len(text_value),
                "secondary_chars": 0,
                "text_chars": len(text_value),
                "markdown_chars": len(markdown),
                "engine_chain": "text",
                "include_secondary_content": False,
                "renderer_backend": "text",
                "renderer_fallback_used": False,
                "renderer_text_recall_ratio": 1.0,
            },
        )

    def _resolve_selected_tags(
        self,
        *,
        options: ExtractContentOptions,
    ) -> set[ExtractContentTag]:
        selected: set[ExtractContentTag]
        if options.include_tags:
            selected = cast("set[ExtractContentTag]", set(options.include_tags))
        else:
            selected = {"body"}
            if options.detail == "full":
                selected.add("sidebar")
        selected -= set(options.exclude_tags)
        return selected

    def _render_document(
        self,
        *,
        snapshot: HtmlSnapshot,
        metadata: TrafilaturaMetadata,
        body_markdown: str,
        options: ExtractContentOptions,
        selected_tags: set[ExtractContentTag],
        base_url: str,
    ) -> RenderBundle:
        blocks: list[str] = []
        stats_list: list[RenderStats] = []
        secondary_markdown = ""
        for semantic_tag in _SEMANTIC_ORDER:
            if semantic_tag not in selected_tags:
                continue
            content, render_stats = self._render_tag(
                snapshot=snapshot,
                metadata=metadata,
                body_markdown=body_markdown,
                semantic_tag=semantic_tag,
                preserve_html_tags=bool(options.include_html_tags),
                base_url=base_url,
            )
            if not content:
                continue
            if semantic_tag == "sidebar":
                secondary_markdown = content
            if semantic_tag not in {"body", "metadata"}:
                stats_list.append(render_stats)
            blocks.append(
                self._wrap_semantic_block(
                    semantic_tag=semantic_tag,
                    content=content,
                    preserve_html_tags=bool(options.include_html_tags),
                )
            )
        return RenderBundle(
            markdown="\n\n".join(part for part in blocks if part).strip(),
            render_stats=merge_render_stats(stats_list),
            secondary_markdown=secondary_markdown,
        )

    def _render_tag(
        self,
        *,
        snapshot: HtmlSnapshot,
        metadata: TrafilaturaMetadata,
        body_markdown: str,
        semantic_tag: ExtractContentTag,
        preserve_html_tags: bool,
        base_url: str,
    ) -> tuple[str, RenderStats]:
        if semantic_tag == "body":
            return body_markdown, empty_render_stats()
        if semantic_tag == "metadata":
            return (
                self._render_metadata_block(
                    metadata=metadata,
                    favicon=snapshot.favicon,
                    preserve_html_tags=preserve_html_tags,
                ),
                empty_render_stats(),
            )
        fragments = list(snapshot.semantic_html.get(semantic_tag, []))
        if not fragments:
            return "", empty_render_stats()
        parts: list[str] = []
        stats_list: list[RenderStats] = []
        for fragment in fragments:
            markdown, render_stats = render_fragment_markdown(
                fragment_html=fragment,
                base_url=base_url,
                preserve_html_tags=preserve_html_tags,
            )
            if markdown:
                parts.append(markdown)
            stats_list.append(render_stats)
        return "\n\n".join(part for part in parts if part).strip(), merge_render_stats(
            stats_list
        )

    def _render_metadata_block(
        self,
        *,
        metadata: TrafilaturaMetadata,
        favicon: str,
        preserve_html_tags: bool,
    ) -> str:
        pairs = [
            ("title", metadata.title),
            ("published_date", metadata.published_date),
            ("author", metadata.author),
            ("image", metadata.image),
            ("favicon", favicon),
        ]
        present_pairs = [(key, value) for key, value in pairs if value]
        if not present_pairs:
            return ""
        if preserve_html_tags:
            items = "".join(
                f"<li><strong>{html.escape(key)}</strong>: {html.escape(value)}</li>"
                for key, value in present_pairs
            )
            return f"<ul>{items}</ul>"
        return "\n".join(f"- {key}: {value}" for key, value in present_pairs)

    def _build_stats(
        self,
        *,
        markdown: str,
        body_markdown: str,
        secondary_markdown: str,
        render_stats: RenderStats,
        detail: Literal["concise", "standard", "full"],
        include_secondary: bool,
        selected_tags: set[ExtractContentTag],
    ) -> StatsMap:
        text_value = markdown_to_text(markdown)
        body_text = markdown_to_text(body_markdown)
        secondary_text = markdown_to_text(secondary_markdown)
        uses_fragment_renderer = any(
            tag in selected_tags
            for tag in ("header", "navigation", "banner", "sidebar", "footer")
        )
        if uses_fragment_renderer:
            renderer_backend = "trafilatura+html_to_markdown"
            engine_chain = "trafilatura+html_to_markdown"
        else:
            renderer_backend = "trafilatura"
            engine_chain = "trafilatura"
        return {
            "engine_chain": engine_chain,
            "content_detail": detail,
            "include_secondary_content": include_secondary,
            "selected_tags": ",".join(sorted(selected_tags)),
            "markdown_chars": len(markdown),
            "text_chars": len(text_value),
            "primary_chars": len(body_text),
            "secondary_chars": len(secondary_text) if include_secondary else 0,
            "renderer_backend": renderer_backend,
            "renderer_fallback_used": False,
            "renderer_text_recall_ratio": float(
                render_stats.get("renderer_text_recall_ratio", 1.0)
            )
            if uses_fragment_renderer
            else 1.0,
            "heading_count": int(render_stats.get("heading_count", 0)),
            "list_count": int(render_stats.get("list_count", 0)),
            "ordered_list_count": int(render_stats.get("ordered_list_count", 0)),
            "table_count": int(render_stats.get("table_count", 0)),
            "table_row_count": int(render_stats.get("table_row_count", 0)),
            "code_block_count": int(render_stats.get("code_block_count", 0)),
            "inline_code_count": int(render_stats.get("inline_code_count", 0)),
            "link_count": int(render_stats.get("link_count", 0)),
            "image_count": int(render_stats.get("image_count", 0)),
            "block_count": int(render_stats.get("block_count", 0)),
        }

    def _build_warnings(
        self,
        *,
        primary_chars: int,
        text_chars: int,
        include_secondary: bool,
    ) -> list[str]:
        warnings: list[str] = []
        if primary_chars < int(self._profile.min_primary_chars):
            warnings.append("trafilatura low primary text")
        if include_secondary:
            if text_chars < int(self._profile.min_total_chars_with_secondary):
                warnings.append("extracted text is short with secondary content")
        elif primary_chars < int(self._profile.min_primary_chars):
            warnings.append("primary content is short")
        return warnings

    def _wrap_semantic_block(
        self,
        *,
        semantic_tag: ExtractContentTag,
        content: str,
        preserve_html_tags: bool,
    ) -> str:
        if preserve_html_tags:
            return (
                f'<section data-serpsage-tag="{semantic_tag}">\n{content}\n</section>'
            )
        if semantic_tag == "body":
            return content
        if semantic_tag == "sidebar":
            return f"## Secondary Content\n\n{content}"
        return f"## {semantic_tag.capitalize()}\n\n{content}"

    def _attach_abstract_markdown(self, doc: ExtractedDocument) -> ExtractedDocument:
        return doc.model_copy(
            update={
                "md_for_abstract": markdown_to_abstract_text(str(doc.markdown or ""))
            }
        )

    def _assert_filtered_output_nonempty(
        self,
        *,
        markdown: str,
        min_text_chars: int,
    ) -> None:
        text_chars = len(clean_whitespace(markdown_to_text(markdown)))
        if not markdown.strip() or text_chars < max(1, int(min_text_chars)):
            raise ValueError("filtered content empty after cleanup")

    def _prepend_title_heading(
        self,
        *,
        markdown: str,
        title: str,
        include_html_tags: bool,
        max_chars: int,
    ) -> str:
        title_norm = clean_whitespace(title)
        if not title_norm:
            return markdown
        compact = clean_whitespace(markdown)
        if title_norm.casefold() in compact[: max(240, len(title_norm) * 6)].casefold():
            return markdown
        heading = (
            f"<h1>{html.escape(title_norm)}</h1>"
            if include_html_tags
            else f"# {title_norm}"
        )
        merged = f"{heading}\n\n{markdown.strip()}".strip()
        return finalize_markdown(markdown=merged, max_chars=max_chars)


def build_extract_profile(*, settings: AppSettings) -> ExtractProfile:
    cfg = settings.fetch.extract
    max_markdown = int(max(8_000, cfg.max_markdown_chars))
    return ExtractProfile(
        max_markdown_chars=max_markdown,
        max_html_chars=max(max_markdown * 3, _MIN_HTML_CAPTURE_CHARS),
        min_text_chars=max(120, int(cfg.min_text_chars)),
        min_primary_chars=max(120, int(cfg.min_primary_chars)),
        min_total_chars_with_secondary=max(
            120,
            int(cfg.min_total_chars_with_secondary),
        ),
        link_max_count=max(1, int(cfg.link_max_count)),
        link_keep_hash=bool(cfg.link_keep_hash),
    )


__all__ = ["HtmlExtractor", "build_extract_profile"]
