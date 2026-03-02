from __future__ import annotations

import html
import importlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from typing_extensions import override
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4.element import Tag

from serpsage.components.extract.base import ExtractorBase
from serpsage.components.extract.markdown.dom import (
    cleanup_dom,
    is_descendant_of,
    is_noise_container,
    is_secondary_container,
    parse_html_document,
    score_primary_candidate,
    text_len,
)
from serpsage.components.extract.markdown.postprocess import (
    finalize_markdown,
    markdown_to_abstract_text,
    markdown_to_text,
)
from serpsage.components.extract.markdown.render import (
    render_markdown,
    render_secondary_markdown,
)
from serpsage.components.extract.utils import (
    decode_best_effort,
    guess_apparent_encoding,
)
from serpsage.components.fetch.utils import classify_content_kind
from serpsage.models.extract import (
    ExtractContentOptions,
    ExtractContentTag,
    ExtractedDocument,
    ExtractedImageLink,
    ExtractedLink,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import AppSettings
trafilatura: Any | None = None
try:
    trafilatura = importlib.import_module("trafilatura")
except Exception:  # noqa: BLE001
    trafilatura = None
StatValue: TypeAlias = int | float | str | bool
StatsMap: TypeAlias = dict[str, StatValue]
SectionName: TypeAlias = Literal["primary", "secondary"]


@dataclass(slots=True)
class SectionBuckets:
    primary_root: Tag | BeautifulSoup
    secondary_roots: list[Tag]


@dataclass(slots=True)
class CandidateDoc:
    markdown: str
    extractor_used: str
    warnings: list[str]
    stats: StatsMap
    primary_chars: int = 0
    secondary_chars: int = 0
    links: list[ExtractedLink] = field(default_factory=list)


@dataclass(slots=True)
class ExtractProfile:
    max_markdown_chars: int
    max_html_chars: int
    min_text_chars: int
    min_primary_chars: int
    min_total_chars_with_secondary: int
    include_secondary_default: bool
    collect_links_default: bool
    link_max_count: int
    link_keep_hash: bool


_SEMANTIC_ORDER: tuple[ExtractContentTag, ...] = (
    "metadata",
    "header",
    "navigation",
    "banner",
    "body",
    "sidebar",
    "footer",
)
_BANNER_HINT_RE = re.compile(r"(banner|hero|masthead|topbar)", re.IGNORECASE)
_TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "msclkid",
}
_SIGNAL_NOISE_RE = re.compile(
    r"(bibliographic|citation|references?|related papers|recommenders?|"
    r"connected papers|bookmark|demos|sciencecast|litmaps|alphaxiv|catalyzex|"
    r"about arxivlabs|code, data and media|tools?)",
    re.IGNORECASE,
)
_PRIMARY_ROOT_SELECTORS: tuple[str, ...] = (
    "main",
    "article",
    '[role="main"]',
    '[itemprop="articleBody"]',
    '[data-testid="article-body"]',
    "#content",
    "#main",
    ".content",
    ".article",
    ".post-content",
    ".entry-content",
)
_MIN_HTML_CAPTURE_CHARS = 5_000_000


class MarkdownExtractor(ExtractorBase):
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
            content_type=content_type, url=url, content=content
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
            doc = self._extract_text(
                text=decoded,
                include_secondary_content=(options.detail == "full"),
            )
            return self._attach_abstract_markdown(doc)
        if kind != "html":
            doc = ExtractedDocument(
                content_kind="binary",
                extractor_used="binary",
                warnings=["unsupported binary content"],
                stats={"primary_chars": 0, "secondary_chars": 0},
            )
            return self._attach_abstract_markdown(doc)
        html_doc = decoded[: int(profile.max_html_chars)]
        if content_options is None:
            if options.detail == "full":
                doc = self._extract_html_full_page(
                    html_doc=html_doc,
                    url=url,
                    profile=profile,
                    collect_links=bool(collect_links),
                    collect_images=bool(collect_images),
                )
                return self._attach_abstract_markdown(doc)
            doc = self._extract_html(
                html_doc=html_doc,
                url=url,
                profile=profile,
                include_secondary_content=False,
                collect_links=bool(collect_links),
                collect_images=bool(collect_images),
            )
            return self._attach_abstract_markdown(doc)
        doc = self._extract_html_with_options(
            html_doc=html_doc,
            url=url,
            profile=profile,
            options=options,
            collect_links=bool(collect_links),
            collect_images=bool(collect_images),
        )
        return self._attach_abstract_markdown(doc)

    def _extract_text(
        self,
        *,
        text: str,
        include_secondary_content: bool,
    ) -> ExtractedDocument:
        lines = [
            cleaned for line in text.splitlines() if (cleaned := clean_whitespace(line))
        ]
        markdown = finalize_markdown(
            markdown="\n\n".join(lines),
            max_chars=int(self._profile.max_markdown_chars),
        )
        text_value = markdown_to_text(markdown)
        stats: StatsMap = {
            "renderer_backend": "text",
            "renderer_fallback_used": False,
            "renderer_text_recall_ratio": 1.0,
        }
        return ExtractedDocument(
            markdown=markdown,
            title="",
            content_kind="text",
            extractor_used="text",
            warnings=[],
            stats={
                "primary_chars": len(text_value),
                "secondary_chars": 0,
                "engine_chain": "text",
                "include_secondary_content": bool(include_secondary_content),
                **stats,
            },
        )

    def _extract_html_with_options(
        self,
        *,
        html_doc: str,
        url: str,
        profile: ExtractProfile,
        options: ExtractContentOptions,
        collect_links: bool,
        collect_images: bool,
    ) -> ExtractedDocument:
        needs_semantic_render = bool(
            options.include_html_tags or options.include_tags or options.exclude_tags
        )
        if options.detail == "full" and not needs_semantic_render:
            return self._extract_html_full_page(
                html_doc=html_doc,
                url=url,
                profile=profile,
                collect_links=collect_links,
                collect_images=collect_images,
            )
        low_doc = self._extract_html(
            html_doc=html_doc,
            url=url,
            profile=profile,
            include_secondary_content=False,
            collect_links=collect_links,
            collect_images=collect_images,
        )
        include_secondary = False
        base_doc = low_doc
        if options.detail == "full":
            include_secondary = True
            base_doc = self._extract_html(
                html_doc=html_doc,
                url=url,
                profile=profile,
                include_secondary_content=True,
                collect_links=collect_links,
                collect_images=collect_images,
            )
        elif options.detail == "standard":
            primary_chars = int(low_doc.stats.get("primary_chars", 0))
            has_non_body_intent = any(
                tag not in {"body", "metadata"} for tag in options.include_tags
            )
            if primary_chars < int(profile.min_primary_chars) or has_non_body_intent:
                include_secondary = True
                base_doc = self._extract_html(
                    html_doc=html_doc,
                    url=url,
                    profile=profile,
                    include_secondary_content=True,
                    collect_links=collect_links,
                    collect_images=collect_images,
                )
        if not needs_semantic_render:
            return base_doc
        return self._extract_html_semantic(
            html_doc=html_doc,
            url=url,
            profile=profile,
            options=options,
            include_secondary=include_secondary,
            collect_links=collect_links,
            collect_images=collect_images,
            base_doc=base_doc,
        )

    def _extract_html_semantic(
        self,
        *,
        html_doc: str,
        url: str,
        profile: ExtractProfile,
        options: ExtractContentOptions,
        include_secondary: bool,
        collect_links: bool,
        collect_images: bool,
        base_doc: ExtractedDocument,
    ) -> ExtractedDocument:
        selected_tags = self._resolve_selected_tags(
            options=options,
            include_secondary=include_secondary,
        )
        keep_tags = (
            {str(tag) for tag in selected_tags}
            | {str(tag) for tag in options.include_tags}
            | {str(tag) for tag in options.exclude_tags}
        )
        soup = parse_html_document(html_doc)
        cleanup_dom(soup, keep_semantic_tags=keep_tags)
        buckets = split_sections(soup=soup, min_primary_chars=profile.min_primary_chars)
        markdown, render_stats = self._render_selected_tags_markdown(
            soup=soup,
            buckets=buckets,
            base_url=url,
            selected_tags=selected_tags,
            preserve_html_tags=bool(options.include_html_tags),
            profile=profile,
        )
        markdown = finalize_markdown(
            markdown=markdown, max_chars=profile.max_markdown_chars
        )
        title = clean_whitespace(
            html.unescape(soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        markdown = self._prepend_title_heading(
            markdown=markdown,
            title=title,
            include_html_tags=bool(options.include_html_tags),
            max_chars=profile.max_markdown_chars,
        )
        text_value = markdown_to_text(markdown)
        self._assert_filtered_output_nonempty(
            markdown=markdown,
            min_text_chars=profile.min_text_chars,
        )
        merged_stats: StatsMap = dict(base_doc.stats)
        merged_stats.update(
            {
                "markdown_chars": len(markdown),
                "text_chars": len(text_value),
                "primary_chars": int(len(text_value)),
                "secondary_chars": int(
                    base_doc.stats.get("secondary_chars", 0)
                    if "sidebar" in selected_tags
                    else 0
                ),
                "include_secondary_content": bool(include_secondary),
                "content_detail": str(options.detail),
                "include_html_tags": bool(options.include_html_tags),
                "selected_tags": ",".join(sorted(selected_tags)),
                "renderer_backend": str(
                    render_stats.get(
                        "renderer_backend",
                        base_doc.stats.get("renderer_backend", "markdownify"),
                    )
                ),
                "renderer_fallback_used": bool(
                    render_stats.get(
                        "renderer_fallback_used",
                        base_doc.stats.get("renderer_fallback_used", False),
                    )
                ),
                "renderer_text_recall_ratio": float(
                    render_stats.get(
                        "renderer_text_recall_ratio",
                        base_doc.stats.get("renderer_text_recall_ratio", 1.0),
                    )
                ),
            }
        )
        fallback_reason = str(render_stats.get("renderer_fallback_reason", "")).strip()
        if fallback_reason:
            merged_stats["renderer_fallback_reason"] = fallback_reason
        merged_warnings = list(base_doc.warnings or [])
        if not text_value.strip() and selected_tags:
            merged_warnings.append("selected tags produced empty content")
        if bool(render_stats.get("renderer_fallback_used", False)):
            merged_warnings.append("renderer fallback used")
        links = (
            collect_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=("sidebar" in selected_tags),
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_links
            else []
        )
        image_links = (
            collect_image_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=("sidebar" in selected_tags),
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_images
            else []
        )
        title = clean_whitespace(
            html.unescape(soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        return ExtractedDocument(
            markdown=markdown,
            title=title,
            content_kind="html",
            extractor_used=str(base_doc.extractor_used or "fastdom"),
            warnings=merged_warnings,
            stats=merged_stats,
            links=links,
            image_links=image_links,
        )

    def _resolve_selected_tags(
        self,
        *,
        options: ExtractContentOptions,
        include_secondary: bool,
    ) -> set[ExtractContentTag]:
        selected: set[ExtractContentTag]
        if options.include_tags:
            selected = set(options.include_tags)
        else:
            selected = {"body"}
            if include_secondary:
                selected.add("sidebar")
        selected -= set(options.exclude_tags)
        return selected

    def _render_selected_tags_markdown(
        self,
        *,
        soup: BeautifulSoup,
        buckets: SectionBuckets,
        base_url: str,
        selected_tags: set[ExtractContentTag],
        preserve_html_tags: bool,
        profile: ExtractProfile,
    ) -> tuple[str, StatsMap]:
        out: list[str] = []
        merged_stats: StatsMap = {
            "renderer_backend": "markdownify",
            "renderer_fallback_used": False,
            "renderer_text_recall_ratio": 1.0,
            "renderer_fallback_reason": "",
        }
        ratio_sum = 0.0
        ratio_count = 0
        for tag in _SEMANTIC_ORDER:
            if tag not in selected_tags:
                continue
            block, block_stats = self._render_semantic_block(
                soup=soup,
                buckets=buckets,
                base_url=base_url,
                semantic_tag=tag,
                preserve_html_tags=preserve_html_tags,
                profile=profile,
            )
            if not block:
                continue
            merged_stats["renderer_fallback_used"] = bool(
                bool(merged_stats.get("renderer_fallback_used", False))
                or bool(block_stats.get("renderer_fallback_used", False))
            )
            ratio_sum += float(block_stats.get("renderer_text_recall_ratio", 1.0))
            ratio_count += 1
            reason = str(block_stats.get("renderer_fallback_reason", "")).strip()
            if reason:
                prev_reason = str(
                    merged_stats.get("renderer_fallback_reason", "")
                ).strip()
                if prev_reason:
                    merged_stats["renderer_fallback_reason"] = ",".join(
                        sorted({*prev_reason.split(","), reason})
                    )
                else:
                    merged_stats["renderer_fallback_reason"] = reason
            if preserve_html_tags:
                out.append(f'<section data-serpsage-tag="{tag}">\n{block}\n</section>')
                continue
            if tag == "sidebar":
                out.append(f"## Secondary Content\n\n{block}")
                continue
            if tag == "body":
                out.append(block)
                continue
            out.append(f"## {tag.capitalize()}\n\n{block}")
        merged_stats["renderer_text_recall_ratio"] = (
            float(ratio_sum / float(ratio_count)) if ratio_count > 0 else 1.0
        )
        return "\n\n".join(out).strip(), merged_stats

    def _render_semantic_block(
        self,
        *,
        soup: BeautifulSoup,
        buckets: SectionBuckets,
        base_url: str,
        semantic_tag: ExtractContentTag,
        preserve_html_tags: bool,
        profile: ExtractProfile,
    ) -> tuple[str, StatsMap]:
        if semantic_tag == "metadata":
            return self._render_metadata_block(
                soup=soup,
                preserve_html_tags=preserve_html_tags,
            ), {
                "renderer_backend": "markdownify",
                "renderer_fallback_used": False,
                "renderer_text_recall_ratio": 1.0,
                "renderer_fallback_reason": "",
            }
        if semantic_tag == "body":
            body_md, body_stats = render_markdown(
                root=buckets.primary_root,
                base_url=base_url,
                skip_roots=buckets.secondary_roots,
                preserve_html_tags=preserve_html_tags,
            )
            return body_md, dict(body_stats)
        if semantic_tag == "sidebar":
            md, secondary_stats = render_secondary_markdown(
                secondary_roots=buckets.secondary_roots,
                base_url=base_url,
                preserve_html_tags=preserve_html_tags,
            )
            return md, dict(secondary_stats)
        roots = self._find_semantic_roots(soup=soup, semantic_tag=semantic_tag)
        if not roots:
            return "", {
                "renderer_backend": "markdownify",
                "renderer_fallback_used": False,
                "renderer_text_recall_ratio": 1.0,
                "renderer_fallback_reason": "",
            }
        parts: list[str] = []
        ratio_sum = 0.0
        ratio_count = 0
        merged_backend = "markdownify"
        fallback_used = False
        reasons: list[str] = []
        for root in roots:
            md, stats = render_markdown(
                root=root,
                base_url=base_url,
                preserve_html_tags=preserve_html_tags,
            )
            if md:
                parts.append(md)
            ratio_sum += float(stats.get("renderer_text_recall_ratio", 1.0))
            ratio_count += 1
            fallback_used = bool(
                fallback_used or bool(stats.get("renderer_fallback_used", False))
            )
            reason = str(stats.get("renderer_fallback_reason", "")).strip()
            if reason:
                reasons.append(reason)
        merged_stats: StatsMap = {
            "renderer_backend": merged_backend,
            "renderer_fallback_used": bool(fallback_used),
            "renderer_text_recall_ratio": (
                float(ratio_sum / float(ratio_count)) if ratio_count > 0 else 1.0
            ),
            "renderer_fallback_reason": ",".join(sorted(set(reasons)))
            if reasons
            else "",
        }
        return "\n\n".join(part for part in parts if part).strip(), merged_stats

    def _render_metadata_block(
        self,
        *,
        soup: BeautifulSoup,
        preserve_html_tags: bool,
    ) -> str:
        title = clean_whitespace(
            html.unescape(soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        meta_pairs: list[tuple[str, str]] = []
        for meta in soup.find_all("meta"):
            key = clean_whitespace(
                str(meta.get("name") or meta.get("property") or "").strip()
            )
            value = clean_whitespace(str(meta.get("content") or "").strip())
            if not key or not value:
                continue
            meta_pairs.append((key, value))
            if len(meta_pairs) >= 20:
                break
        if preserve_html_tags:
            lines: list[str] = []
            if title:
                lines.append(f"<title>{html.escape(title)}</title>")
            for key, value in meta_pairs:
                lines.append(
                    f'<meta name="{html.escape(key, quote=True)}" '
                    f'content="{html.escape(value, quote=True)}" />'
                )
            return "\n".join(lines).strip()
        lines = []
        if title:
            lines.append(f"- title: {title}")
        for key, value in meta_pairs:
            lines.append(f"- {key}: {value}")
        return "\n".join(lines).strip()

    def _find_semantic_roots(
        self,
        *,
        soup: BeautifulSoup,
        semantic_tag: ExtractContentTag,
    ) -> list[Tag]:
        roots: list[Tag] = []
        if semantic_tag == "header":
            roots.extend(soup.find_all("header"))
        elif semantic_tag == "navigation":
            roots.extend(soup.find_all("nav"))
            roots.extend(soup.select('[role="navigation"]'))
        elif semantic_tag == "banner":
            roots.extend(soup.select('[role="banner"]'))
            for node in soup.find_all(True):
                ident = " ".join(
                    [
                        str(node.get("id") or ""),
                        " ".join(node.get("class") or []),
                    ]
                ).strip()
                if ident and _BANNER_HINT_RE.search(ident):
                    roots.append(node)
        elif semantic_tag == "footer":
            roots.extend(soup.find_all("footer"))
            roots.extend(soup.select('[role="contentinfo"]'))
        deduped: list[Tag] = []
        for root in roots:
            if any(root is keep for keep in deduped):
                continue
            if any(root in keep.descendants for keep in deduped):
                continue
            deduped.append(root)
        return deduped

    def _extract_html(
        self,
        *,
        html_doc: str,
        url: str,
        profile: ExtractProfile,
        include_secondary_content: bool,
        collect_links: bool,
        collect_images: bool,
    ) -> ExtractedDocument:
        soup = parse_html_document(html_doc)
        cleanup_dom(soup)
        buckets = split_sections(soup=soup, min_primary_chars=profile.min_primary_chars)
        best = run_fastdom(
            html_doc=html_doc,
            buckets=buckets,
            profile=profile,
            base_url=url,
            include_secondary_content=include_secondary_content,
        )
        markdown = finalize_markdown(
            markdown=best.markdown,
            max_chars=profile.max_markdown_chars,
        )
        title = clean_whitespace(
            html.unescape(soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        markdown = self._prepend_title_heading(
            markdown=markdown,
            title=title,
            include_html_tags=False,
            max_chars=profile.max_markdown_chars,
        )
        text_value = markdown_to_text(markdown)
        self._assert_filtered_output_nonempty(
            markdown=markdown,
            min_text_chars=profile.min_text_chars,
        )
        merged_stats: StatsMap = dict(best.stats)
        merged_stats.update(
            {
                "markdown_chars": len(markdown),
                "text_chars": len(text_value),
                "candidate_count": 1,
                "engine_chain": "fastdom",
                "include_secondary_content": bool(include_secondary_content),
                "primary_chars": int(best.primary_chars),
                "secondary_chars": int(
                    best.secondary_chars if include_secondary_content else 0
                ),
                "renderer_backend": str(
                    best.stats.get("renderer_backend", "markdownify")
                ),
                "renderer_fallback_used": bool(
                    best.stats.get("renderer_fallback_used", False)
                ),
                "renderer_text_recall_ratio": float(
                    best.stats.get("renderer_text_recall_ratio", 1.0)
                ),
            }
        )
        fallback_reason = str(best.stats.get("renderer_fallback_reason", "")).strip()
        if fallback_reason:
            merged_stats["renderer_fallback_reason"] = fallback_reason
        merged_warnings = list(
            dict.fromkeys(
                str(item).strip() for item in best.warnings if str(item).strip()
            )
        )
        if include_secondary_content:
            if len(text_value) < int(profile.min_total_chars_with_secondary):
                merged_warnings.append("extracted text is short with secondary content")
        elif int(best.primary_chars) < int(profile.min_primary_chars):
            merged_warnings.append("primary content is short")
        if bool(best.stats.get("renderer_fallback_used", False)):
            merged_warnings.append("renderer fallback used")
        merged_warnings = list(dict.fromkeys(merged_warnings))
        links = (
            collect_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=include_secondary_content,
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_links
            else []
        )
        image_links = (
            collect_image_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=include_secondary_content,
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_images
            else []
        )
        return ExtractedDocument(
            markdown=markdown,
            title=title,
            content_kind="html",
            extractor_used=str(best.extractor_used or "fastdom"),
            warnings=merged_warnings,
            stats=merged_stats,
            links=links,
            image_links=image_links,
        )

    def _extract_html_full_page(
        self,
        *,
        html_doc: str,
        url: str,
        profile: ExtractProfile,
        collect_links: bool,
        collect_images: bool,
    ) -> ExtractedDocument:
        soup = parse_html_document(html_doc)
        cleanup_dom(soup)
        root = soup.body if soup.body is not None else soup
        markdown, render_stats = render_markdown(
            root=root,
            base_url=url,
            preserve_html_tags=False,
        )
        markdown = finalize_markdown(
            markdown=markdown,
            max_chars=profile.max_markdown_chars,
        )
        title = clean_whitespace(
            html.unescape(soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        markdown = self._prepend_title_heading(
            markdown=markdown,
            title=title,
            include_html_tags=False,
            max_chars=profile.max_markdown_chars,
        )
        text_value = markdown_to_text(markdown)
        self._assert_filtered_output_nonempty(
            markdown=markdown,
            min_text_chars=profile.min_text_chars,
        )
        stats: StatsMap = {
            "engine_chain": "full_page",
            "content_detail": "full",
            "include_secondary_content": True,
            "markdown_chars": len(markdown),
            "text_chars": len(text_value),
            "primary_chars": len(text_value),
            "secondary_chars": 0,
            "renderer_backend": str(
                render_stats.get("renderer_backend", "markdownify")
            ),
            "renderer_fallback_used": bool(
                render_stats.get("renderer_fallback_used", False)
            ),
            "renderer_text_recall_ratio": float(
                render_stats.get("renderer_text_recall_ratio", 1.0)
            ),
        }
        fallback_reason = str(render_stats.get("renderer_fallback_reason", "")).strip()
        if fallback_reason:
            stats["renderer_fallback_reason"] = fallback_reason
        warnings: list[str] = []
        if bool(render_stats.get("renderer_fallback_used", False)):
            warnings.append("renderer fallback used")
        buckets = split_sections(soup=soup, min_primary_chars=profile.min_primary_chars)
        links = (
            collect_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=True,
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_links
            else []
        )
        image_links = (
            collect_image_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=True,
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_images
            else []
        )
        return ExtractedDocument(
            markdown=markdown,
            title=title,
            content_kind="html",
            extractor_used="full_page",
            warnings=warnings,
            stats=stats,
            links=links,
            image_links=image_links,
        )

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
        if title_norm.lower() in compact[: max(240, len(title_norm) * 6)].lower():
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
        # Large modern pages (for example GitHub repos) can place meaningful content
        # well after 480k chars; keep at least the fetch-layer HTML budget here.
        max_html_chars=max(max_markdown * 3, _MIN_HTML_CAPTURE_CHARS),
        min_text_chars=max(120, int(cfg.min_text_chars)),
        min_primary_chars=max(120, int(cfg.min_primary_chars)),
        min_total_chars_with_secondary=max(
            120,
            int(cfg.min_total_chars_with_secondary),
        ),
        include_secondary_default=bool(cfg.include_secondary_content_default),
        collect_links_default=bool(cfg.collect_links_default),
        link_max_count=max(1, int(cfg.link_max_count)),
        link_keep_hash=bool(cfg.link_keep_hash),
    )


def split_sections(*, soup: BeautifulSoup, min_primary_chars: int) -> SectionBuckets:
    primary = _pick_primary_root(soup=soup, min_primary_chars=min_primary_chars)
    secondary = _collect_secondary_roots(soup=soup, primary_root=primary)
    secondary = _dedupe_nested_roots(secondary)
    return SectionBuckets(primary_root=primary, secondary_roots=secondary)


def _pick_primary_root(
    *, soup: BeautifulSoup, min_primary_chars: int
) -> Tag | BeautifulSoup:
    min_seed_chars = max(80, int(min_primary_chars) // 3)
    for selector in _PRIMARY_ROOT_SELECTORS:
        node = soup.select_one(selector)
        if node is not None and text_len(node) >= min_seed_chars:
            return node
    best: Tag | BeautifulSoup = soup
    best_score = -1.0
    for cand in soup.find_all(["article", "main", "section", "div"]):
        if is_noise_container(cand):
            continue
        score = score_primary_candidate(cand)
        if score > best_score:
            best_score = score
            best = cand
    return best


def _collect_secondary_roots(
    *,
    soup: BeautifulSoup,
    primary_root: Tag | BeautifulSoup,
) -> list[Tag]:
    out: list[Tag] = []
    for node in soup.find_all(True):
        if node is primary_root:
            continue
        if is_noise_container(node):
            continue
        if not is_secondary_container(node):
            continue
        if text_len(node) < 20:
            continue
        if is_descendant_of(node, primary_root):
            continue
        out.append(node)
    return out


def _dedupe_nested_roots(roots: list[Tag]) -> list[Tag]:
    deduped: list[Tag] = []
    for node in roots:
        if any(is_descendant_of(node, keep) for keep in deduped):
            continue
        deduped.append(node)
    return deduped


def collect_links_inventory(
    *,
    soup: BeautifulSoup,
    base_url: str,
    buckets: SectionBuckets,
    include_secondary_content: bool,
    max_links: int,
    keep_hash: bool,
) -> list[ExtractedLink]:
    out: list[ExtractedLink] = []
    seen: set[tuple[str, str, SectionName]] = set()
    base_norm = _normalize_url(url=base_url, base_url=base_url, keep_hash=keep_hash)
    if base_norm is None:
        base_norm = base_url
    base_netloc = urlparse(base_norm).netloc.lower()
    position = 0
    for anchor in soup.find_all("a"):
        position += 1
        href = str(anchor.get("href") or "").strip()
        text = clean_whitespace(anchor.get_text(" ", strip=True))
        if not href or not text:
            continue
        normalized = _normalize_url(url=href, base_url=base_url, keep_hash=keep_hash)
        if not normalized:
            continue
        section = _section_for_tag(anchor, buckets=buckets)
        if section == "secondary" and not include_secondary_content:
            continue
        key = (normalized, text.lower(), section)
        if key in seen:
            continue
        seen.add(key)
        parsed = urlparse(normalized)
        rel = {str(x).lower() for x in (anchor.get("rel") or [])}
        same_page = _strip_fragment(normalized) == _strip_fragment(base_norm)
        out.append(
            ExtractedLink(
                url=normalized,
                anchor_text=text,
                section=section,
                is_internal=(parsed.netloc.lower() == base_netloc)
                if parsed.netloc
                else True,
                nofollow=("nofollow" in rel),
                same_page=same_page,
                source_hint=_source_hint(anchor),
                position=position,
            )
        )
        if len(out) >= max(1, int(max_links)):
            break
    return out


def collect_image_links_inventory(
    *,
    soup: BeautifulSoup,
    base_url: str,
    buckets: SectionBuckets,
    include_secondary_content: bool,
    max_links: int,
    keep_hash: bool,
) -> list[ExtractedImageLink]:
    out: list[ExtractedImageLink] = []
    seen: set[tuple[str, SectionName]] = set()
    base_norm = _normalize_url(url=base_url, base_url=base_url, keep_hash=keep_hash)
    if base_norm is None:
        base_norm = base_url
    base_netloc = urlparse(base_norm).netloc.lower()
    position = 0
    for image in soup.find_all(["img", "source"]):
        position += 1
        section = _section_for_tag(image, buckets=buckets)
        if section == "secondary" and not include_secondary_content:
            continue
        alt_text = clean_whitespace(str(image.get("alt") or "").strip())
        candidates = _image_url_candidates(image=image)
        for raw in candidates:
            normalized = _normalize_url(
                url=raw,
                base_url=base_url,
                keep_hash=keep_hash,
            )
            if not normalized:
                continue
            key = (normalized, section)
            if key in seen:
                continue
            seen.add(key)
            parsed = urlparse(normalized)
            out.append(
                ExtractedImageLink(
                    url=normalized,
                    alt_text=alt_text,
                    section=section,
                    is_internal=(parsed.netloc.lower() == base_netloc)
                    if parsed.netloc
                    else True,
                    source_hint=_source_hint(image),
                    position=position,
                )
            )
            if len(out) >= max(1, int(max_links)):
                return out
    return out


def _normalize_url(*, url: str, base_url: str, keep_hash: bool) -> str | None:
    raw = (url or "").strip()
    if not raw:
        return None
    if raw.lower().startswith(("javascript:", "mailto:", "tel:", "data:")):
        return None
    try:
        joined = str(urljoin(base_url, raw))
        parsed = urlparse(joined)
        if parsed.scheme not in {"http", "https"}:
            return None
        clean_pairs = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_KEYS and not k.lower().startswith("utm_")
        ]
        fragment = parsed.fragment if keep_hash else ""
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                urlencode(clean_pairs, doseq=True),
                fragment,
            )
        )
    except Exception:
        return None


def _strip_fragment(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )


def _section_for_tag(tag: Tag, *, buckets: SectionBuckets) -> SectionName:
    for sec in buckets.secondary_roots:
        if tag is sec or is_descendant_of(tag, sec):
            return "secondary"
    if tag is buckets.primary_root or is_descendant_of(tag, buckets.primary_root):
        return "primary"
    return "secondary"


def _source_hint(tag: Tag) -> str:
    cur: Tag | None = tag
    hops = 0
    while cur is not None and hops < 4:
        ident = " ".join(
            [str(cur.get("id") or ""), " ".join(cur.get("class") or [])]
        ).strip()
        if ident:
            return ident[:120]
        parent = cur.parent
        if not isinstance(parent, Tag):
            break
        cur = parent
        hops += 1
    return "unknown"


def _image_url_candidates(*, image: Tag) -> list[str]:
    out: list[str] = []
    src = str(image.get("src") or "").strip()
    if src:
        out.append(src)
    srcset = str(image.get("srcset") or "").strip()
    if srcset:
        for item in srcset.split(","):
            url = item.strip().split(" ", 1)[0].strip()
            if url:
                out.append(url)
    data_src = str(image.get("data-src") or "").strip()
    if data_src:
        out.append(data_src)
    data_srcset = str(image.get("data-srcset") or "").strip()
    if data_srcset:
        for item in data_srcset.split(","):
            url = item.strip().split(" ", 1)[0].strip()
            if url:
                out.append(url)
    return out


def run_fastdom(
    *,
    html_doc: str,
    buckets: SectionBuckets,
    profile: ExtractProfile,
    base_url: str,
    include_secondary_content: bool,
) -> CandidateDoc:
    if trafilatura is None:
        raise RuntimeError("trafilatura is required by the fastdom extractor")
    secondary_md = ""
    secondary_stats: dict[str, int | float | bool | str] = {
        "renderer_text_recall_ratio": 1.0,
        "renderer_fallback_used": False,
    }
    if include_secondary_content and buckets.secondary_roots:
        secondary_md, secondary_stats = render_secondary_markdown(
            secondary_roots=buckets.secondary_roots,
            base_url=base_url,
        )
    try:
        signal_text = (
            trafilatura.extract(
                html_doc,
                output_format="txt",
                include_tables=True,
                include_links=False,
                include_formatting=False,
                favor_precision=True,
                favor_recall=False,
            )
            or ""
        )
    except Exception as exc:  # noqa: BLE001
        return CandidateDoc(
            markdown="",
            extractor_used="fastdom",
            warnings=[f"trafilatura_failed:{type(exc).__name__}"],
            stats={
                "engine": "fastdom",
                "renderer_backend": "trafilatura",
                "renderer_fallback_used": False,
                "renderer_text_recall_ratio": 0.0,
            },
            primary_chars=0,
            secondary_chars=0,
        )
    primary_md, primary_stats = render_markdown(
        root=buckets.primary_root,
        base_url=base_url,
        skip_roots=buckets.secondary_roots,
    )
    primary_md, localized, localization_reason = _clip_markdown_with_signal(
        markdown=primary_md,
        signal_text=signal_text,
        min_chars=profile.min_primary_chars,
    )
    markdown = primary_md.strip()
    if include_secondary_content and secondary_md.strip():
        markdown = (
            f"{markdown}\n\n## Secondary Content\n\n{secondary_md.strip()}".strip()
        )
    markdown = finalize_markdown(
        markdown=markdown, max_chars=profile.max_markdown_chars
    )
    text_value = markdown_to_text(markdown)
    primary_chars = len(markdown_to_text(primary_md))
    secondary_chars = len(markdown_to_text(secondary_md)) if secondary_md else 0
    primary_ratio = float(primary_stats.get("renderer_text_recall_ratio", 1.0))
    secondary_ratio = float(secondary_stats.get("renderer_text_recall_ratio", 1.0))
    ratio_den = max(1, primary_chars + secondary_chars)
    blended_ratio = float(
        primary_ratio * primary_chars + secondary_ratio * secondary_chars
    ) / float(ratio_den)
    fallback_used = bool(primary_stats.get("renderer_fallback_used", False)) or bool(
        secondary_stats.get("renderer_fallback_used", False)
    )
    stats: StatsMap = {
        "engine": "fastdom",
        "primary_chars": int(primary_chars),
        "secondary_chars": int(secondary_chars),
        "renderer_backend": str(primary_stats.get("renderer_backend", "markdownify")),
        "renderer_fallback_used": bool(fallback_used),
        "renderer_text_recall_ratio": float(blended_ratio),
        "signal_chars": int(len(clean_whitespace(signal_text))),
        "signal_localized": bool(localized),
    }
    if localization_reason:
        stats["signal_localization_reason"] = localization_reason
    fallback_reason_parts: list[str] = []
    primary_reason = str(primary_stats.get("renderer_fallback_reason", "")).strip()
    if primary_reason:
        fallback_reason_parts.append(primary_reason)
    secondary_reason = str(secondary_stats.get("renderer_fallback_reason", "")).strip()
    if secondary_reason:
        fallback_reason_parts.append(secondary_reason)
    if fallback_reason_parts:
        stats["renderer_fallback_reason"] = ",".join(sorted(set(fallback_reason_parts)))
    warnings: list[str] = []
    if primary_chars < int(profile.min_primary_chars):
        warnings.append("fastdom low primary text")
    if len(text_value) < int(profile.min_text_chars):
        warnings.append("fastdom low text output")
    if fallback_used:
        warnings.append("renderer fallback used")
    return CandidateDoc(
        markdown=markdown,
        extractor_used="fastdom",
        warnings=warnings,
        stats=stats,
        primary_chars=primary_chars,
        secondary_chars=secondary_chars,
    )


def _clip_markdown_with_signal(
    *,
    markdown: str,
    signal_text: str,
    min_chars: int,
) -> tuple[str, bool, str]:
    if not markdown.strip():
        return markdown, False, "markdown_empty"
    signal_norm = clean_whitespace(signal_text)
    if len(signal_norm) < 120:
        return markdown, False, "signal_short"
    anchors = _pick_signal_anchors(signal_text)
    if anchors is None:
        return markdown, False, "signal_anchors_missing"
    start_anchor, end_anchor = anchors
    blocks = _markdown_blocks(markdown)
    if len(blocks) < 4:
        return markdown, False, "markdown_too_short_for_clip"
    block_plain = [clean_whitespace(markdown_to_text(block)) for block in blocks]
    stitched = "\n\n".join(block_plain).lower()
    start_pos = _find_anchor_position(stitched, start_anchor)
    if start_pos < 0:
        return markdown, False, "start_anchor_not_found"
    end_pos = _find_anchor_position(stitched, end_anchor)
    if end_pos < start_pos:
        end_pos = start_pos + max(20, len(clean_whitespace(start_anchor)))
    starts: list[int] = []
    cursor = 0
    for txt in block_plain:
        starts.append(cursor)
        cursor += len(txt) + 2
    start_idx = 0
    for idx, st in enumerate(starts):
        if st + len(block_plain[idx]) >= start_pos:
            start_idx = max(0, idx - 1)
            break
    end_idx = len(blocks) - 1
    for idx, st in enumerate(starts):
        if st > end_pos:
            end_idx = min(len(blocks) - 1, idx)
            break
    while end_idx > start_idx and _looks_like_noise_block(block_plain[end_idx]):
        end_idx -= 1
    clipped = "\n\n".join(blocks[start_idx : end_idx + 1]).strip()
    if not clipped:
        return markdown, False, "clip_empty"
    plain_full = clean_whitespace(markdown_to_text(markdown))
    plain_clipped = clean_whitespace(markdown_to_text(clipped))
    if len(plain_clipped) < max(int(min_chars), int(len(plain_full) * 0.45)):
        return markdown, False, "clip_too_short"
    if len(plain_clipped) >= int(len(plain_full) * 0.985):
        return markdown, False, "clip_not_effective"
    return clipped, True, "signal_localized"


def _pick_signal_anchors(signal_text: str) -> tuple[str, str] | None:
    paragraphs = [
        cleaned
        for part in re.split(
            r"\n\s*\n", signal_text.replace("\r\n", "\n").replace("\r", "\n")
        )
        if (cleaned := clean_whitespace(part))
    ]
    candidates = [p for p in paragraphs if _is_content_paragraph(p)]
    if len(candidates) < 2:
        return None
    start = candidates[0]
    end = candidates[-1]
    for para in reversed(candidates):
        if not _looks_like_noise_block(para):
            end = para
            break
    start_key = clean_whitespace(start).lower()
    end_key = clean_whitespace(end).lower()
    if start_key == end_key:
        return None
    return start, end


def _is_content_paragraph(text: str) -> bool:
    normalized = clean_whitespace(text)
    if len(normalized) < 60:
        return False
    if len(normalized.split()) < 8:
        return False
    punct = len(re.findall(r"[,.!?;:\u3002\uff01\uff1f\uff1b]", normalized))
    return punct >= 2


def _markdown_blocks(markdown: str) -> list[str]:
    source = markdown.replace("\r\n", "\n").replace("\r", "\n")
    return [blk.strip() for blk in re.split(r"\n{2,}", source) if blk.strip()]


def _find_anchor_position(haystack: str, anchor: str) -> int:
    target = clean_whitespace(anchor).lower()
    variants: list[str] = []
    if target:
        variants.append(target)
    for n in (180, 120, 90, 64):
        if len(target) > n:
            variants.append(target[:n])  # noqa: PERF401
    variants = [v for i, v in enumerate(variants) if v and v not in variants[:i]]
    for variant in variants:
        pos = haystack.find(variant)
        if pos >= 0:
            return pos
    return -1


def _looks_like_noise_block(text: str) -> bool:
    normalized = clean_whitespace(text)
    if not normalized:
        return True
    if _SIGNAL_NOISE_RE.search(normalized):
        return True
    punct = len(re.findall(r"[,.!?;:\u3002\uff01\uff1f\uff1b]", normalized))
    words = len(normalized.split())
    return bool(words <= 12 and punct == 0)
