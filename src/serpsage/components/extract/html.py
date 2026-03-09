from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, cast
from typing_extensions import override
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import trafilatura
from html_to_markdown import (
    ConversionOptions,
    MetadataConfig,
    PreprocessingOptions,
    TableExtractionResult,
    convert_with_tables,
)
from selectolax.parser import HTMLParser, Node

from serpsage.components.base import ComponentMeta
from serpsage.components.extract.base import ExtractConfigBase, ExtractorBase
from serpsage.components.extract.utils import (
    decode_best_effort,
    finalize_markdown,
    guess_apparent_encoding,
    markdown_to_text,
)
from serpsage.components.fetch.utils import classify_content_kind
from serpsage.components.registry import register_component
from serpsage.models.components.extract import (
    ExtractContent,
    ExtractContentTag,
    ExtractedDocument,
    ExtractMeta,
    ExtractRef,
    ExtractRefs,
    ExtractSpec,
    ExtractTrace,
)
from serpsage.utils import clean_whitespace

_HTML_EXTRACTOR_META = ComponentMeta(
    family="extract",
    name="html",
    version="1.0.0",
    summary="HTML and text content extractor.",
    provides=("extract.html_engine",),
    config_model=ExtractConfigBase,
)


@register_component(meta=_HTML_EXTRACTOR_META)
class HtmlExtractor(ExtractorBase):
    meta = _HTML_EXTRACTOR_META

    @dataclass(slots=True)
    class Profile:
        max_markdown_chars: int
        max_html_chars: int
        min_text_chars: int
        min_primary_chars: int
        min_total_chars_with_secondary: int
        link_max_count: int
        link_keep_hash: bool

    @dataclass(slots=True)
    class Snapshot:
        tree: HTMLParser
        primary_path: tuple[int, ...] | None
        secondary_paths: list[tuple[int, ...]]
        semantic_html: dict[ExtractContentTag, list[str]]
        favicon: str

    @dataclass(slots=True)
    class TrafilaturaResult:
        title: str = ""
        published_date: str = ""
        author: str = ""
        image: str = ""
        markdown: str = ""

    @dataclass(slots=True)
    class RenderBundle:
        markdown: str
        secondary_markdown: str
        stats: dict[str, int | float | str | bool]

    _SEMANTIC_ORDER: ClassVar[tuple[ExtractContentTag, ...]] = (
        "metadata",
        "header",
        "navigation",
        "banner",
        "body",
        "sidebar",
        "footer",
    )
    _MIN_HTML_CAPTURE_CHARS: ClassVar[int] = 5_000_000
    _TRACKING_KEYS: ClassVar[set[str]] = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "msclkid",
    }
    _SECONDARY_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(sidebar|related|recommend|comment|discussion|thread|reply|faq|"
        r"supplement|appendix|toc|table-of-contents|index|more-like-this)",
        re.IGNORECASE,
    )
    _BANNER_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(banner|hero|masthead|topbar)", re.IGNORECASE
    )
    _BLOCK_TAGS: ClassVar[set[str]] = {
        "article",
        "aside",
        "blockquote",
        "caption",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
    _COUNT_KEYS: ClassVar[tuple[str, ...]] = (
        "heading_count",
        "list_count",
        "ordered_list_count",
        "table_count",
        "table_row_count",
        "code_block_count",
        "inline_code_count",
        "link_count",
        "image_count",
        "block_count",
    )
    _CONVERSION_OPTIONS: ClassVar[ConversionOptions] = ConversionOptions(
        heading_style="atx",
        list_indent_type="spaces",
        list_indent_width=2,
        bullets="-*+",
        strong_em_symbol="*",
        escape_asterisks=False,
        escape_underscores=False,
        escape_misc=False,
        extract_metadata=False,
        whitespace_mode="normalized",
        wrap=False,
        newline_style="spaces",
        code_block_style="backticks",
    )
    _PREPROCESSING: ClassVar[PreprocessingOptions] = PreprocessingOptions(enabled=False)
    _METADATA_CONFIG: ClassVar[MetadataConfig] = MetadataConfig(
        extract_document=False,
        extract_headers=True,
        extract_links=True,
        extract_images=True,
        extract_structured_data=False,
    )

    def __init__(self) -> None:
        self._profile = self._build_profile()

    @override
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        content_options: ExtractSpec | None = None,
        include_secondary_content: bool = False,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        kind = classify_content_kind(
            content_type=content_type,
            url=url,
            content=content,
        )
        if kind == "pdf":
            raise ValueError(
                "HtmlExtractor does not handle PDF; use AutoExtractor/PdfExtractor"
            )
        decoded, decoded_kind = decode_best_effort(
            content,
            content_type=content_type,
            apparent_encoding=guess_apparent_encoding(content),
        )
        if kind == "unknown":
            kind = "html" if decoded_kind == "html" else "text"
        options = content_options or ExtractSpec(
            detail="full" if include_secondary_content else "concise"
        )
        if kind == "text":
            return self._finalize_content(
                doc=self._extract_text(decoded),
                content_options=options,
            )
        if kind != "html":
            return self._finalize_content(
                doc=ExtractedDocument(
                    trace=ExtractTrace(
                        kind="binary",
                        engine="binary",
                        warnings=["unsupported binary content"],
                        stats={"primary_chars": 0, "secondary_chars": 0},
                    )
                ),
                content_options=options,
            )
        raw_html = decoded[: self._profile.max_html_chars]
        snapshot = self._snapshot(raw_html=raw_html, base_url=url)
        extracted = self._extract_trafilatura(
            raw_html=raw_html,
            url=url,
            max_chars=self._profile.max_markdown_chars,
        )
        selected_tags = self._resolve_tags(options)
        rendered = self._render_document(
            snapshot=snapshot,
            extracted=extracted,
            selected_tags=selected_tags,
            base_url=url,
            preserve_html_tags=bool(options.keep_html),
        )
        markdown = finalize_markdown(
            markdown=rendered.markdown,
            max_chars=self._profile.max_markdown_chars,
        )
        markdown = self._prepend_title(
            markdown=markdown,
            title=extracted.title,
            include_html_tags=bool(options.keep_html),
        )
        self._assert_nonempty(markdown=markdown)
        include_secondary = "sidebar" in selected_tags
        stats = self._stats(
            markdown=markdown,
            body_markdown=extracted.markdown,
            secondary_markdown=rendered.secondary_markdown,
            render_stats=rendered.stats,
            detail=options.detail,
            include_secondary=include_secondary,
            selected_tags=selected_tags,
        )
        return self._finalize_content(
            doc=ExtractedDocument(
                content=ExtractContent(markdown=markdown),
                meta=ExtractMeta(
                    title=extracted.title,
                    published_date=extracted.published_date,
                    author=extracted.author,
                    image=extracted.image,
                    favicon=snapshot.favicon,
                ),
                refs=ExtractRefs(
                    links=(
                        self._links(
                            snapshot=snapshot,
                            base_url=url,
                            include_secondary=include_secondary,
                        )
                        if collect_links
                        else []
                    ),
                    images=(
                        self._images(
                            snapshot=snapshot,
                            base_url=url,
                            include_secondary=include_secondary,
                        )
                        if collect_images
                        else []
                    ),
                ),
                trace=ExtractTrace(
                    kind="html",
                    engine=str(stats["engine_chain"]),
                    warnings=self._warnings(
                        primary_chars=int(stats["primary_chars"]),
                        text_chars=int(stats["text_chars"]),
                        include_secondary=include_secondary,
                    ),
                    stats=stats,
                ),
            ),
            content_options=options,
        )

    def _extract_text(self, text: str) -> ExtractedDocument:
        markdown = finalize_markdown(
            markdown="\n\n".join(
                cleaned
                for line in text.splitlines()
                if (cleaned := clean_whitespace(line))
            ),
            max_chars=self._profile.max_markdown_chars,
        )
        plain = markdown_to_text(markdown)
        return ExtractedDocument(
            content=ExtractContent(markdown=markdown),
            trace=ExtractTrace(
                kind="text",
                engine="text",
                stats={
                    "primary_chars": len(plain),
                    "secondary_chars": 0,
                    "text_chars": len(plain),
                    "markdown_chars": len(markdown),
                    "engine_chain": "text",
                    "include_secondary_content": False,
                    "renderer_backend": "text",
                    "renderer_fallback_used": False,
                    "renderer_text_recall_ratio": 1.0,
                },
            ),
        )

    def _build_profile(self) -> Profile:
        cfg = self.config
        max_markdown = int(max(8_000, cfg.max_markdown_chars))
        return self.Profile(
            max_markdown_chars=max_markdown,
            max_html_chars=max(max_markdown * 3, self._MIN_HTML_CAPTURE_CHARS),
            min_text_chars=max(120, int(cfg.min_text_chars)),
            min_primary_chars=max(120, int(cfg.min_primary_chars)),
            min_total_chars_with_secondary=max(
                120, int(cfg.min_total_chars_with_secondary)
            ),
            link_max_count=max(1, int(cfg.link_max_count)),
            link_keep_hash=bool(cfg.link_keep_hash),
        )

    @classmethod
    def _resolve_tags(cls, options: ExtractSpec) -> set[ExtractContentTag]:
        if options.sections:
            return cast("set[ExtractContentTag]", set(options.sections))
        return {"body", "sidebar"} if options.detail == "full" else {"body"}

    @classmethod
    def _snapshot(cls, *, raw_html: str, base_url: str) -> Snapshot:
        tree = HTMLParser(raw_html)
        primary = next(
            (
                max(nodes, key=cls._text_len)
                for selector in ("main", "article", "[role='main']", "body")
                if (
                    nodes := [
                        node for node in tree.css(selector) if cls._node_text(node)
                    ]
                )
            ),
            tree.body if tree.body is not None and cls._node_text(tree.body) else None,
        )
        secondary = cls._dedupe_nodes(
            [
                node
                for node in [
                    *tree.css("aside, [role='complementary']"),
                    *[
                        node
                        for selector in (
                            "[class*='sidebar']",
                            "[id*='sidebar']",
                            "[class*='related']",
                            "[id*='related']",
                            "[class*='recommend']",
                            "[id*='recommend']",
                            "[class*='comment']",
                            "[id*='comment']",
                            "[class*='discussion']",
                            "[id*='discussion']",
                            "[class*='faq']",
                            "[id*='faq']",
                            "[class*='appendix']",
                            "[id*='appendix']",
                        )
                        for node in tree.css(selector)
                    ],
                ]
                if cls._is_secondary(node)
                and (primary is None or cls._node_id(node) != cls._node_id(primary))
            ]
        )
        semantic_html: dict[ExtractContentTag, list[str]] = {
            "metadata": [],
            "header": cls._unique_html(tree.css("header")),
            "navigation": cls._unique_html(tree.css("nav, [role='navigation']")),
            "banner": cls._unique_html(
                [
                    node
                    for selector in (
                        "[role='banner']",
                        "[class*='banner']",
                        "[id*='banner']",
                        "[class*='hero']",
                        "[id*='hero']",
                        "[class*='masthead']",
                        "[id*='masthead']",
                    )
                    for node in tree.css(selector)
                    if cls._is_banner(node)
                ]
            ),
            "body": [str(primary.html or "")] if primary is not None else [],
            "sidebar": [str(node.html or "") for node in secondary if node.html],
            "footer": cls._unique_html(tree.css("footer, [role='contentinfo']")),
        }
        return cls.Snapshot(
            tree=tree,
            primary_path=cls._node_path(primary),
            secondary_paths=[
                path for node in secondary if (path := cls._node_path(node)) is not None
            ],
            semantic_html=semantic_html,
            favicon=cls._favicon(tree=tree, base_url=base_url),
        )

    @classmethod
    def _extract_trafilatura(
        cls,
        *,
        raw_html: str,
        url: str,
        max_chars: int,
    ) -> TrafilaturaResult:
        raw = trafilatura.bare_extraction(
            raw_html,
            url=url,
            include_comments=False,
            output_format="python",
            favor_precision=True,
            favor_recall=False,
            include_tables=True,
            include_images=False,
            include_formatting=True,
            include_links=True,
            deduplicate=False,
            with_metadata=True,
        )
        payload: dict[str, Any] | None
        if isinstance(raw, dict):
            payload = raw
        elif raw is not None and callable(getattr(raw, "as_dict", None)):
            payload = cast("dict[str, Any]", raw.as_dict())
        else:
            payload = None
        if payload is None:
            raise ValueError("trafilatura bare_extraction returned invalid payload")
        markdown = finalize_markdown(
            markdown=str(payload.get("text") or payload.get("raw_text") or "").strip(),
            max_chars=max_chars,
        )
        if not markdown:
            raise ValueError("trafilatura bare_extraction returned empty markdown")
        image = clean_whitespace(str(payload.get("image") or ""))
        return cls.TrafilaturaResult(
            title=clean_whitespace(str(payload.get("title") or "")),
            published_date=clean_whitespace(str(payload.get("date") or "")),
            author=clean_whitespace(str(payload.get("author") or "")),
            image=urljoin(url, image) if image else "",
            markdown=markdown,
        )

    @classmethod
    def _render_document(
        cls,
        *,
        snapshot: Snapshot,
        extracted: TrafilaturaResult,
        selected_tags: set[ExtractContentTag],
        base_url: str,
        preserve_html_tags: bool,
    ) -> RenderBundle:
        blocks: list[str] = []
        secondary_markdown = ""
        stats_items: list[dict[str, int | float | str | bool]] = []
        for tag in cls._SEMANTIC_ORDER:
            if tag not in selected_tags:
                continue
            if tag == "body":
                content = extracted.markdown
                stats = cls._empty_render_stats()
            elif tag == "metadata":
                content = cls._metadata_block(
                    extracted=extracted,
                    favicon=snapshot.favicon,
                    preserve_html_tags=preserve_html_tags,
                )
                stats = cls._empty_render_stats()
            else:
                rendered = [
                    cls._render_fragment(
                        fragment_html=fragment,
                        base_url=base_url,
                        preserve_html_tags=preserve_html_tags,
                    )
                    for fragment in snapshot.semantic_html.get(tag, [])
                ]
                content = "\n\n".join(
                    markdown for markdown, _ in rendered if markdown
                ).strip()
                stats = cls._merge_render_stats(
                    [item_stats for _, item_stats in rendered]
                )
            if not content:
                continue
            if tag == "sidebar":
                secondary_markdown = content
            if tag not in {"body", "metadata"}:
                stats_items.append(stats)
            blocks.append(cls._wrap_block(tag, content, preserve_html_tags))
        return cls.RenderBundle(
            markdown="\n\n".join(blocks).strip(),
            secondary_markdown=secondary_markdown,
            stats=cls._merge_render_stats(stats_items),
        )

    @classmethod
    def _render_fragment(
        cls,
        *,
        fragment_html: str,
        base_url: str,
        preserve_html_tags: bool,
    ) -> tuple[str, dict[str, int | float | str | bool]]:
        tree = HTMLParser(fragment_html)
        for node in tree.css(
            "[href], [src], [poster], [data-src], [srcset], [data-srcset]"
        ):
            for attr in ("href", "src", "poster", "data-src"):
                raw = str(node.attributes.get(attr, "")).strip()
                if raw:
                    node.attributes[attr] = urljoin(base_url, raw)
            for attr in ("srcset", "data-srcset"):
                raw = str(node.attributes.get(attr, "")).strip()
                if raw:
                    node.attributes[attr] = ", ".join(
                        (
                            f"{urljoin(base_url, item.split(' ', 1)[0].strip())} {item.split(' ', 1)[1].strip()}".strip()
                            if " " in item
                            else urljoin(base_url, item)
                        )
                        for candidate in raw.split(",")
                        if (item := candidate.strip())
                    )
        prepared = (
            str(tree.body.html or "")
            if tree.body is not None
            else str(tree.html or fragment_html)
        )
        result = convert_with_tables(
            prepared,
            options=replace(
                cls._CONVERSION_OPTIONS,
                preserve_tags=(
                    {
                        str(node.tag).lower()
                        for node in HTMLParser(prepared).css("*")
                        if str(node.tag or "").strip()
                    }
                    if preserve_html_tags
                    else None
                ),
            )
            if preserve_html_tags
            else cls._CONVERSION_OPTIONS,
            preprocessing=cls._PREPROCESSING,
            metadata_config=cls._METADATA_CONFIG,
        )
        markdown = "\n".join(
            line.rstrip()
            for line in str(result["content"])
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
            .split("\n")
        ).strip()
        if markdown:
            markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        return markdown, cls._render_stats(
            fragment_html=prepared,
            markdown=markdown,
            result=result,
        )

    @classmethod
    def _metadata_block(
        cls,
        *,
        extracted: TrafilaturaResult,
        favicon: str,
        preserve_html_tags: bool,
    ) -> str:
        pairs = [
            ("title", extracted.title),
            ("published_date", extracted.published_date),
            ("author", extracted.author),
            ("image", extracted.image),
            ("favicon", favicon),
        ]
        present = [(key, value) for key, value in pairs if value]
        if not present:
            return ""
        if preserve_html_tags:
            return (
                "<ul>"
                + "".join(
                    f"<li><strong>{html_lib.escape(key)}</strong>: {html_lib.escape(value)}</li>"
                    for key, value in present
                )
                + "</ul>"
            )
        return "\n".join(f"- {key}: {value}" for key, value in present)

    @classmethod
    def _wrap_block(
        cls, tag: ExtractContentTag, content: str, preserve_html_tags: bool
    ) -> str:
        if preserve_html_tags:
            return f'<section data-serpsage-tag="{tag}">\n{content}\n</section>'
        if tag == "body":
            return content
        if tag == "sidebar":
            return f"## Secondary Content\n\n{content}"
        return f"## {tag.capitalize()}\n\n{content}"

    def _stats(
        self,
        *,
        markdown: str,
        body_markdown: str,
        secondary_markdown: str,
        render_stats: dict[str, int | float | str | bool],
        detail: Literal["concise", "standard", "full"],
        include_secondary: bool,
        selected_tags: set[ExtractContentTag],
    ) -> dict[str, int | float | str | bool]:
        text_value = markdown_to_text(markdown)
        body_text = markdown_to_text(body_markdown)
        secondary_text = markdown_to_text(secondary_markdown)
        use_fragment_renderer = any(
            tag in selected_tags
            for tag in ("header", "navigation", "banner", "sidebar", "footer")
        )
        backend = (
            "trafilatura+html_to_markdown" if use_fragment_renderer else "trafilatura"
        )
        return {
            "engine_chain": backend,
            "content_detail": detail,
            "include_secondary_content": include_secondary,
            "selected_tags": ",".join(sorted(selected_tags)),
            "markdown_chars": len(markdown),
            "text_chars": len(text_value),
            "primary_chars": len(body_text),
            "secondary_chars": len(secondary_text) if include_secondary else 0,
            "renderer_backend": backend,
            "renderer_fallback_used": False,
            "renderer_text_recall_ratio": float(
                render_stats.get("renderer_text_recall_ratio", 1.0)
            )
            if use_fragment_renderer
            else 1.0,
            **{key: int(render_stats.get(key, 0)) for key in self._COUNT_KEYS},
        }

    def _warnings(
        self, *, primary_chars: int, text_chars: int, include_secondary: bool
    ) -> list[str]:
        warnings: list[str] = []
        if primary_chars < self._profile.min_primary_chars:
            warnings.append("trafilatura low primary text")
        if include_secondary:
            if text_chars < self._profile.min_total_chars_with_secondary:
                warnings.append("extracted text is short with secondary content")
        elif primary_chars < self._profile.min_primary_chars:
            warnings.append("primary content is short")
        return warnings

    def _links(
        self, *, snapshot: Snapshot, base_url: str, include_secondary: bool
    ) -> list[ExtractRef]:
        base_norm = (
            self._normalize_url(
                url=base_url,
                base_url=base_url,
                keep_hash=self._profile.link_keep_hash,
            )
            or base_url
        )
        base_netloc = urlparse(base_norm).netloc.lower()
        items: list[ExtractRef] = []
        seen: set[tuple[str, str, str]] = set()
        for position, anchor in enumerate(snapshot.tree.css("a"), start=1):
            href = str(anchor.attributes.get("href", "")).strip()
            text = self._node_text(anchor)
            if not href or not text:
                continue
            normalized = self._normalize_url(
                url=href,
                base_url=base_url,
                keep_hash=self._profile.link_keep_hash,
            )
            if not normalized:
                continue
            section = self._section(anchor=anchor, snapshot=snapshot)
            if section == "secondary" and not include_secondary:
                continue
            key = (normalized, text.casefold(), section)
            if key in seen:
                continue
            seen.add(key)
            parsed = urlparse(normalized)
            rel = str(anchor.attributes.get("rel", "")).strip().lower()
            items.append(
                ExtractRef(
                    url=normalized,
                    text=text,
                    zone=cast("Literal['primary', 'secondary']", section),
                    internal=(parsed.netloc.lower() == base_netloc)
                    if parsed.netloc
                    else True,
                    nofollow=("nofollow" in rel.split()),
                    same_page=(
                        self._strip_fragment(normalized)
                        == self._strip_fragment(base_norm)
                    ),
                    source=self._source_hint(anchor),
                    position=position,
                )
            )
            if len(items) >= self._profile.link_max_count:
                break
        return items

    def _images(
        self, *, snapshot: Snapshot, base_url: str, include_secondary: bool
    ) -> list[ExtractRef]:
        base_norm = (
            self._normalize_url(
                url=base_url,
                base_url=base_url,
                keep_hash=self._profile.link_keep_hash,
            )
            or base_url
        )
        base_netloc = urlparse(base_norm).netloc.lower()
        items: list[ExtractRef] = []
        seen: set[tuple[str, str]] = set()
        for position, image in enumerate(snapshot.tree.css("img, source"), start=1):
            section = self._section(anchor=image, snapshot=snapshot)
            if section == "secondary" and not include_secondary:
                continue
            alt_text = clean_whitespace(str(image.attributes.get("alt", "")).strip())
            for candidate in self._image_candidates(image):
                normalized = self._normalize_url(
                    url=candidate,
                    base_url=base_url,
                    keep_hash=self._profile.link_keep_hash,
                )
                if not normalized:
                    continue
                key = (normalized, section)
                if key in seen:
                    continue
                seen.add(key)
                parsed = urlparse(normalized)
                items.append(
                    ExtractRef(
                        url=normalized,
                        text=alt_text,
                        zone=cast("Literal['primary', 'secondary']", section),
                        internal=(parsed.netloc.lower() == base_netloc)
                        if parsed.netloc
                        else True,
                        source=self._source_hint(image),
                        position=position,
                    )
                )
                if len(items) >= self._profile.link_max_count:
                    return items
        return items

    def _prepend_title(
        self, *, markdown: str, title: str, include_html_tags: bool
    ) -> str:
        title_norm = clean_whitespace(title)
        if not title_norm:
            return markdown
        lead = clean_whitespace(markdown)[: max(240, len(title_norm) * 6)].casefold()
        if title_norm.casefold() in lead:
            return markdown
        heading = (
            f"<h1>{html_lib.escape(title_norm)}</h1>"
            if include_html_tags
            else f"# {title_norm}"
        )
        return finalize_markdown(
            markdown=f"{heading}\n\n{markdown.strip()}".strip(),
            max_chars=self._profile.max_markdown_chars,
        )

    def _assert_nonempty(self, *, markdown: str) -> None:
        text_chars = len(clean_whitespace(markdown_to_text(markdown)))
        if not markdown.strip() or text_chars < max(1, self._profile.min_text_chars):
            raise ValueError("filtered content empty after cleanup")

    @classmethod
    def _empty_render_stats(cls) -> dict[str, int | float | str | bool]:
        return {
            "heading_count": 0,
            "list_count": 0,
            "ordered_list_count": 0,
            "table_count": 0,
            "table_row_count": 0,
            "code_block_count": 0,
            "inline_code_count": 0,
            "link_count": 0,
            "image_count": 0,
            "block_count": 0,
            "renderer_backend": "html_to_markdown",
            "renderer_fallback_used": False,
            "renderer_text_recall_ratio": 1.0,
        }

    @classmethod
    def _merge_render_stats(
        cls, items: list[dict[str, int | float | str | bool]]
    ) -> dict[str, int | float | str | bool]:
        if not items:
            return cls._empty_render_stats()
        merged = cls._empty_render_stats()
        ratio_sum = 0.0
        for item in items:
            for key in cls._COUNT_KEYS:
                merged[key] = int(merged.get(key, 0)) + int(item.get(key, 0))
            ratio_sum += float(item.get("renderer_text_recall_ratio", 1.0))
        merged["renderer_text_recall_ratio"] = ratio_sum / float(len(items))
        return merged

    @classmethod
    def _render_stats(
        cls,
        *,
        fragment_html: str,
        markdown: str,
        result: TableExtractionResult,
    ) -> dict[str, int | float | str | bool]:
        tree = HTMLParser(fragment_html)
        metadata = cast("dict[str, Any]", result.get("metadata") or {})
        tables = cast("list[dict[str, Any]]", result.get("tables", []))
        source_text = tree.text(separator=" ", strip=True)
        source_chars = len(clean_whitespace(source_text))
        plain = markdown_to_text(markdown)
        return {
            "heading_count": len(cast("list[Any]", metadata.get("headers", []))),
            "list_count": len(tree.css("li")),
            "ordered_list_count": sum(
                1
                for item in tree.css("li")
                if item.parent is not None
                and str(item.parent.tag or "").lower() == "ol"
            ),
            "table_count": len(tables),
            "table_row_count": sum(
                1
                for table in tables
                for is_header in cast("list[bool]", table.get("is_header_row", []))
                if not bool(is_header)
            ),
            "code_block_count": len(tree.css("pre")),
            "inline_code_count": sum(
                1
                for node in tree.css("code, kbd, samp")
                if not cls._has_pre_ancestor(node)
            ),
            "link_count": len(cast("list[Any]", metadata.get("links", []))),
            "image_count": len(cast("list[Any]", metadata.get("images", []))),
            "block_count": cls._block_count(fragment_html),
            "renderer_backend": "html_to_markdown",
            "renderer_fallback_used": False,
            "renderer_text_recall_ratio": (
                1.0
                if source_chars <= 0
                else max(0.0, min(1.0, float(len(plain)) / float(source_chars)))
            ),
        }

    @classmethod
    def _unique_html(cls, nodes: list[Node]) -> list[str]:
        seen: set[str] = set()
        items: list[str] = []
        for node in cls._dedupe_nodes(nodes):
            html_value = str(node.html or "").strip()
            if not html_value or html_value in seen:
                continue
            seen.add(html_value)
            items.append(html_value)
        return items

    @classmethod
    def _dedupe_nodes(cls, nodes: list[Node]) -> list[Node]:
        unique = {
            path: node for node in nodes if (path := cls._node_path(node)) is not None
        }
        kept: list[tuple[tuple[int, ...], Node]] = []
        for path, node in sorted(unique.items(), key=lambda item: len(item[0])):
            if any(cls._within(path, kept_path) for kept_path, _ in kept):
                continue
            kept.append((path, node))
        return [node for _, node in kept]

    @classmethod
    def _section(cls, *, anchor: Node, snapshot: Snapshot) -> str:
        path = cls._node_path(anchor)
        if path is None:
            return "secondary"
        if any(cls._within(path, secondary) for secondary in snapshot.secondary_paths):
            return "secondary"
        if snapshot.primary_path is not None and cls._within(
            path, snapshot.primary_path
        ):
            return "primary"
        return "secondary"

    @classmethod
    def _favicon(cls, *, tree: HTMLParser, base_url: str) -> str:
        for node in tree.css("link[rel]"):
            rel = str(node.attributes.get("rel", "")).strip().lower()
            href = str(node.attributes.get("href", "")).strip()
            if "icon" in rel and href:
                normalized = cls._normalize_url(
                    url=href,
                    base_url=base_url,
                    keep_hash=True,
                )
                if normalized:
                    return normalized
        return ""

    @classmethod
    def _normalize_url(cls, *, url: str, base_url: str, keep_hash: bool) -> str | None:
        raw = (url or "").strip()
        if not raw or raw.lower().startswith(
            ("javascript:", "mailto:", "tel:", "data:")
        ):
            return None
        try:
            parsed = urlparse(urljoin(base_url, raw))
        except Exception:
            return None
        if parsed.scheme not in {"http", "https"}:
            return None
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                urlencode(
                    [
                        (key, value)
                        for key, value in parse_qsl(
                            parsed.query, keep_blank_values=True
                        )
                        if key.lower() not in cls._TRACKING_KEYS
                        and not key.lower().startswith("utm_")
                    ],
                    doseq=True,
                ),
                parsed.fragment if keep_hash else "",
            )
        )

    @classmethod
    def _strip_fragment(cls, url: str) -> str:
        parsed = urlparse(url)
        return urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, "")
        )

    @classmethod
    def _source_hint(cls, node: Node) -> str:
        current: Node | None = node
        for _ in range(4):
            if current is None:
                break
            ident = " ".join(
                [
                    str(current.attributes.get("id", "")).strip(),
                    str(current.attributes.get("class", "")).strip(),
                ]
            ).strip()
            if ident:
                return ident[:120]
            current = current.parent
        return "unknown"

    @classmethod
    def _image_candidates(cls, image: Node) -> list[str]:
        values = [
            str(image.attributes.get(attr, "")).strip()
            for attr in ("src", "data-src")
            if str(image.attributes.get(attr, "")).strip()
        ]
        values.extend(
            item.strip().split(" ", 1)[0].strip()
            for attr in ("srcset", "data-srcset")
            if str(image.attributes.get(attr, "")).strip()
            for item in str(image.attributes.get(attr, "")).split(",")
            if item.strip()
        )
        return values

    @classmethod
    def _is_secondary(cls, node: Node) -> bool:
        ident = " ".join(
            [
                str(node.attributes.get("id", "")).strip(),
                str(node.attributes.get("class", "")).strip(),
            ]
        ).strip()
        style = str(node.attributes.get("style", "")).strip().lower().replace(" ", "")
        return (
            cls._text_len(node) >= 20
            and str(node.attributes.get("aria-hidden", "")).strip().lower() != "true"
            and "display:none" not in style
            and "visibility:hidden" not in style
            and (
                node.tag == "aside"
                or str(node.attributes.get("role", "")).strip().lower()
                == "complementary"
                or bool(ident and cls._SECONDARY_RE.search(ident))
            )
        )

    @classmethod
    def _is_banner(cls, node: Node) -> bool:
        role = str(node.attributes.get("role", "")).strip().lower()
        ident = " ".join(
            [
                str(node.attributes.get("id", "")).strip(),
                str(node.attributes.get("class", "")).strip(),
            ]
        ).strip()
        return role == "banner" or bool(ident and cls._BANNER_RE.search(ident))

    @classmethod
    def _has_pre_ancestor(cls, node: Node) -> bool:
        current = node.parent
        while current is not None:
            if str(current.tag or "").lower() == "pre":
                return True
            current = current.parent
        return False

    @classmethod
    def _block_count(cls, fragment_html: str) -> int:
        return sum(
            1
            for node in HTMLParser(fragment_html).css(",".join(sorted(cls._BLOCK_TAGS)))
            if cls._node_text(node)
        )

    @classmethod
    def _text_len(cls, node: Node | None) -> int:
        return len(cls._node_text(node))

    @classmethod
    def _node_text(cls, node: Node | None) -> str:
        return (
            ""
            if node is None
            else clean_whitespace(node.text(separator=" ", strip=True))
        )

    @classmethod
    def _node_path(cls, node: Node | None) -> tuple[int, ...] | None:
        if node is None:
            return None
        parts: list[int] = []
        current: Node | None = node
        while current is not None and current.parent is not None:
            parent = current.parent
            if parent.child is None:
                return None
            siblings: list[Node] = []
            cursor: Node | None = parent.child
            while cursor is not None:
                siblings.append(cursor)
                cursor = cursor.next
            try:
                parts.append(
                    next(
                        idx
                        for idx, sibling in enumerate(siblings)
                        if cls._node_id(sibling) == cls._node_id(current)
                    )
                )
            except StopIteration:
                return None
            current = parent
            if current.tag == "html":
                break
        parts.reverse()
        return tuple(parts)

    @classmethod
    def _within(cls, path: tuple[int, ...], prefix: tuple[int, ...]) -> bool:
        return len(path) >= len(prefix) and path[: len(prefix)] == prefix

    @classmethod
    def _node_id(cls, node: Node) -> int:
        value = node.mem_id
        return int(value()) if callable(value) else int(value)


__all__ = ["HtmlExtractor"]
