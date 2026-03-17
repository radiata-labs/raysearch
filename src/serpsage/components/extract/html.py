from __future__ import annotations

import html as html_lib
import json
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

from serpsage.components.extract.base import ExtractConfigBase, ExtractorBase
from serpsage.components.extract.utils import (
    decode_best_effort,
    guess_apparent_encoding,
    markdown_to_text,
)
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


class HtmlExtractorConfig(ExtractConfigBase):
    __setting_family__ = "extract"
    __setting_name__ = "html"


class HtmlExtractor(ExtractorBase[HtmlExtractorConfig]):
    """Extract structured content from HTML pages.

    Uses multiple extraction strategies with fallbacks:
    1. Trafilatura on cleaned HTML (noise removed)
    2. Standard Trafilatura extraction
    3. DOM-based extraction from semantic content
    4. Script content fallback (for SPAs)
    """

    _DETAIL_DEFAULT_TAGS: ClassVar[dict[str, tuple[ExtractContentTag, ...]]] = {
        "concise": ("body",),
        "standard": ("header", "body"),
        "full": (
            "header",
            "navigation",
            "banner",
            "body",
            "sidebar",
            "footer",
        ),
    }
    _SEMANTIC_ORDER: ClassVar[tuple[ExtractContentTag, ...]] = (
        "header",
        "navigation",
        "banner",
        "body",
        "sidebar",
        "footer",
    )
    _MIN_HTML_CAPTURE_CHARS: ClassVar[int] = 5_000_000

    # Content detection thresholds
    _MIN_HEURISTIC_TEXT_CHARS: ClassVar[int] = (
        200  # Min chars for heuristic content detection
    )
    _MIN_CONTENT_SCORE_CHARS: ClassVar[int] = 80  # Min chars for content scoring

    # URL normalization
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

    # Content detection patterns
    _SECONDARY_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(sidebar|related|recommend|comment|discussion|thread|reply|faq|"
        r"supplement|appendix|toc|table-of-contents|index|more-like-this)",
        re.IGNORECASE,
    )
    _BANNER_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(banner|hero|masthead|topbar)", re.IGNORECASE
    )

    # HTML block tags for counting
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

    # Render statistics keys
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

    # Markdown conversion options
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

    # Documentation site content selectors
    _DOCS_CONTENT_SELECTORS: ClassVar[tuple[str, ...]] = (
        # VitePress
        ".vp-doc",
        ".vitepress-doc",
        "[class*='vp-doc']",
        # Docusaurus
        "article.markdown",
        "div.markdown",
        "[class*='docItemContainer']",
        "[class*='docMainContainer']",
        # GitBook
        "[class*='gitbook-root']",
        "[data-testid='page.contentEditor']",
        # ReadTheDocs/Sphinx
        "[role='main']",
        ".wy-nav-content",
        ".document",
        # MkDocs
        ".md-content",
        "[class*='md-content']",
        # Notion-based docs
        ".notion-page-content",
        "[class*='notion-page']",
        # General patterns
        "[class*='documentation']",
        "[class*='docs-content']",
        "[class*='content-wrapper']",
        "div[class*='prose']",
        # API docs
        "[class*='redoc']",
        "[class*='swagger-ui']",
        # Fallbacks
        "main article",
        "article main",
        ".content",
        "#content",
    )

    # Noise selectors to remove before extraction
    # These are ALWAYS removed (ads, scripts, etc.)
    _NOISE_SELECTORS_GLOBAL: ClassVar[tuple[str, ...]] = (
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        ".ads",
        ".advertisement",
        "[class*='social-']",
        "[class*='share-']",
        "[class*='ad-']",
        "[class*='ads-']",
        "#ads",
        "#advertisement",
    )

    # Structural noise - only removed when NOT inside main content
    # (handled separately to preserve semantic tags within articles)
    _NOISE_SELECTORS_STRUCTURAL: ClassVar[tuple[str, ...]] = (
        "nav",
        "header",
        "footer",
        "aside",
        "[role='navigation']",
        "[role='banner']",
        "[role='contentinfo']",
        ".nav",
        ".navigation",
        ".sidebar",
        ".header",
        ".footer",
        ".social-share",
        ".related-posts",
        ".comments",
        "#nav",
        "#navigation",
        "#sidebar",
        "#header",
        "#footer",
        "[class*='comment']",
    )

    @dataclass(slots=True)
    class Profile:
        """Extraction profile with configurable limits."""

        max_markdown_chars: int
        max_html_chars: int
        min_text_chars: int
        min_primary_chars: int
        min_total_chars_with_secondary: int
        link_max_count: int
        link_keep_hash: bool

    @dataclass(slots=True)
    class Snapshot:
        """Parsed HTML snapshot with detected content regions."""

        tree: HTMLParser
        primary_path: tuple[int, ...] | None
        secondary_paths: list[tuple[int, ...]]
        semantic_html: dict[ExtractContentTag, list[str]]
        favicon: str

    @dataclass(slots=True)
    class TrafilaturaResult:
        """Result from trafilatura extraction."""

        title: str = ""
        published_date: str = ""
        author: str = ""
        image: str = ""
        markdown: str = ""
        engine: str = "trafilatura"
        description: str = ""
        site_name: str = ""
        locale: str = ""

    @dataclass(slots=True)
    class RenderBundle:
        """Rendered document with statistics."""

        markdown: str
        secondary_markdown: str
        stats: dict[str, int | float | str | bool]

    @dataclass(slots=True)
    class EnrichedMeta:
        """Enhanced metadata from multiple sources (JSON-LD, OpenGraph, Twitter, meta tags)."""

        title: str = ""
        description: str = ""
        author: str = ""
        published_date: str = ""
        modified_date: str = ""
        image: str = ""
        site_name: str = ""
        locale: str = ""
        keywords: list[str] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            if self.keywords is None:
                self.keywords = []

    def __init__(self) -> None:
        self._profile = self._build_profile()

    @override
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        crawl_backend: str = "curl_cffi",
        content_kind: Literal[
            "html", "pdf", "text", "markdown", "json", "binary", "unknown"
        ] = "unknown",
        content_options: ExtractSpec | None = None,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        """Main entry point for HTML content extraction."""
        if content_kind == "pdf":
            raise ValueError(
                "HtmlExtractor does not handle PDF; use AutoExtractor/PdfExtractor"
            )

        decoded, decoded_kind = decode_best_effort(
            content,
            content_type=content_type,
            apparent_encoding=guess_apparent_encoding(content),
        )
        # Use content_kind if known, otherwise infer from decoded content
        if content_kind == "unknown":
            kind = "html" if decoded_kind == "html" else "text"
        else:
            kind = content_kind

        options = content_options or ExtractSpec()

        # Handle markdown content directly (e.g., from Cloudflare edge markdown)
        if kind == "markdown":
            return self._finalize_content(
                doc=self._extract_markdown(decoded),
                content_options=options,
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
        selected_tags = self._resolve_tags(options)
        extracted = self._extract_primary(
            raw_html=raw_html,
            snapshot=snapshot,
            url=url,
            preserve_html_tags=bool(options.keep_html),
        )
        rendered = self._render_document(
            snapshot=snapshot,
            extracted=extracted,
            selected_tags=selected_tags,
            base_url=url,
            preserve_html_tags=bool(options.keep_html),
        )
        markdown = rendered.markdown
        self._assert_nonempty(markdown=markdown)

        include_secondary = "sidebar" in selected_tags
        stats = self._stats(
            markdown=markdown,
            body_markdown=extracted.markdown,
            secondary_markdown=rendered.secondary_markdown,
            render_stats=rendered.stats,
            primary_engine=extracted.engine,
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
                            snapshot=snapshot, base_url=url, include_secondary=True
                        )
                        if collect_links
                        else []
                    ),
                    images=(
                        self._images(
                            snapshot=snapshot, base_url=url, include_secondary=True
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

    # --------------------------------------------------------------------------
    # Text Extraction
    # --------------------------------------------------------------------------

    def _extract_text(self, text: str) -> ExtractedDocument:
        """Extract content from plain text (non-HTML)."""
        markdown = "\n\n".join(
            cleaned for line in text.splitlines() if (cleaned := clean_whitespace(line))
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

    def _extract_markdown(self, markdown: str) -> ExtractedDocument:
        """Extract content from pre-formatted markdown (e.g., Cloudflare edge markdown).

        This is an optimization path - when the server returns markdown directly,
        we skip HTML parsing entirely and use the content as-is.
        """
        # Clean up the markdown while preserving structure
        cleaned = clean_whitespace(markdown)
        plain = markdown_to_text(cleaned)
        return ExtractedDocument(
            content=ExtractContent(markdown=cleaned),
            trace=ExtractTrace(
                kind="markdown",
                engine="markdown_direct",
                stats={
                    "primary_chars": len(plain),
                    "secondary_chars": 0,
                    "text_chars": len(plain),
                    "markdown_chars": len(cleaned),
                    "engine_chain": "markdown_direct",
                    "include_secondary_content": False,
                    "renderer_backend": "markdown",
                    "renderer_fallback_used": False,
                    "renderer_text_recall_ratio": 1.0,
                },
            ),
        )

    # --------------------------------------------------------------------------
    # Profile & Configuration
    # --------------------------------------------------------------------------

    def _build_profile(self) -> Profile:
        """Build extraction profile from configuration."""
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

    def _resolve_tags(self, options: ExtractSpec) -> set[ExtractContentTag]:
        """Resolve content tags based on extraction options."""
        if options.sections:
            selected = set(options.sections)
        else:
            selected = set(self._DETAIL_DEFAULT_TAGS.get(options.detail, ("body",)))
        return selected

    # --------------------------------------------------------------------------
    # HTML Snapshot & Content Detection
    # --------------------------------------------------------------------------

    @classmethod
    def _snapshot(cls, *, raw_html: str, base_url: str) -> Snapshot:
        """Create parsed HTML snapshot with content region detection."""
        tree = HTMLParser(raw_html)
        primary = cls._find_primary_content(tree)

        secondary = cls._dedupe_nodes(
            [
                node
                for node in [
                    *tree.css("aside, [role='complementary']"),
                    *[
                        n
                        for sel in (
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
                        for n in tree.css(sel)
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
                    n
                    for sel in (
                        "[role='banner']",
                        "[class*='banner']",
                        "[id*='banner']",
                        "[class*='hero']",
                        "[id*='hero']",
                        "[class*='masthead']",
                        "[id*='masthead']",
                    )
                    for n in tree.css(sel)
                    if cls._is_banner(n)
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
    def _find_primary_content(cls, tree: HTMLParser) -> Node | None:
        """Find the primary content node using multiple strategies.

        Priority:
        1. Documentation-specific selectors (VitePress, Docusaurus, etc.)
        2. Semantic HTML elements (main, article)
        3. Heuristic-based content detection
        4. Body fallback
        """
        # Strategy 1: Documentation-specific selectors
        for selector in cls._DOCS_CONTENT_SELECTORS:
            nodes = [n for n in tree.css(selector) if cls._node_text(n)]
            if nodes:
                return max(nodes, key=cls._text_len)

        # Strategy 2: Semantic HTML elements
        for selector in ("main", "article", "[role='main']"):
            nodes = [n for n in tree.css(selector) if cls._node_text(n)]
            if nodes:
                return max(nodes, key=cls._text_len)

        # Strategy 3: Heuristic content detection
        content_node = cls._find_content_by_heuristic(tree)
        if content_node:
            return content_node

        # Strategy 4: Body fallback
        if tree.body is not None and cls._node_text(tree.body):
            return tree.body

        return None

    @classmethod
    def _find_content_by_heuristic(cls, tree: HTMLParser) -> Node | None:
        """Find content using text density and structural heuristics."""
        candidates: list[tuple[float, Node]] = []

        for node in tree.css("div, section"):
            text = cls._node_text(node)
            if len(text) < cls._MIN_HEURISTIC_TEXT_CHARS:
                continue

            class_id = cls._node_ident(node).lower()

            # Skip obvious non-content areas
            if any(
                skip in class_id
                for skip in (
                    "nav",
                    "footer",
                    "sidebar",
                    "menu",
                    "header",
                    "comment",
                    "social",
                    "share",
                    "related",
                    "advertisement",
                    "ad-",
                    "ads-",
                )
            ):
                continue

            # Calculate content score
            text_len = len(text)
            p_count = len(node.css("p"))
            h_count = len(node.css("h1, h2, h3, h4, h5, h6"))
            code_count = len(node.css("pre, code"))
            list_count = len(node.css("li"))

            score = float(text_len)
            score += p_count * 50.0  # Paragraphs are strong content signals
            score += h_count * 30.0  # Headings indicate structure
            score += code_count * 40.0  # Code blocks indicate technical content
            score += list_count * 10.0  # Lists can be content

            # Bonus for high text-to-tag ratio (text density)
            html_len = len(str(node.html or ""))
            if html_len > 0:
                density = text_len / html_len
                if density > 0.3:
                    score *= 1.5
                elif density > 0.2:
                    score *= 1.2

            candidates.append((score, node))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # --------------------------------------------------------------------------
    # Primary Extraction Strategies
    # --------------------------------------------------------------------------

    def _extract_primary(
        self,
        *,
        raw_html: str,
        snapshot: Snapshot,
        url: str,
        preserve_html_tags: bool,
    ) -> TrafilaturaResult:
        """Extract primary content using multiple fallback strategies."""
        extracted: HtmlExtractor.TrafilaturaResult | None = None
        failure: ValueError | None = None

        # Strategy 1: Try trafilatura on cleaned HTML (better for noisy pages)
        try:
            extracted = self._extract_trafilatura_cleaned(raw_html=raw_html, url=url)
        except ValueError as exc:
            failure = exc

        # Strategy 2: Standard trafilatura
        if extracted is None:
            try:
                extracted = self._extract_trafilatura(raw_html=raw_html, url=url)
            except ValueError as exc:
                if failure is None:
                    failure = exc

        # Strategy 3: DOM-based extraction from semantic content
        if extracted is None:
            extracted = self._extract_dom_fallback(
                snapshot=snapshot,
                url=url,
                preserve_html_tags=preserve_html_tags,
            )

        # Strategy 4: Script content fallback (for SPAs)
        if extracted is None:
            extracted = self._extract_script_fallback(
                raw_html=raw_html,
                url=url,
                preserve_html_tags=preserve_html_tags,
            )

        if extracted is None:
            raise failure or ValueError("html extraction produced no primary content")

        return self._fill_missing_metadata(
            extracted=extracted, snapshot=snapshot, url=url
        )

    def _extract_trafilatura_cleaned(
        self, *, raw_html: str, url: str
    ) -> TrafilaturaResult | None:
        """Try trafilatura on cleaned HTML (with noise elements removed)."""
        tree = HTMLParser(raw_html)

        # Always remove global noise (scripts, ads, etc.)
        for selector in self._NOISE_SELECTORS_GLOBAL:
            for node in tree.css(selector):
                node.decompose()

        # Find primary content region before removing structural noise
        primary = self._find_primary_content(tree)

        # Remove structural noise only if NOT inside primary content
        for selector in self._NOISE_SELECTORS_STRUCTURAL:
            for node in tree.css(selector):
                # Skip if this node is inside or is the primary content
                if primary is not None and (
                    self._node_id(node) == self._node_id(primary)
                    or self._is_descendant(parent=primary, child=node)
                ):
                    continue
                node.decompose()

        cleaned_html = str(tree.html) if tree.html else ""
        if not cleaned_html:
            return None

        raw = self._trafilatura_extract(cleaned_html, url)
        if raw is None:
            return None

        markdown = str(raw.get("text") or raw.get("raw_text") or "").strip()
        if not markdown:
            return None

        image = clean_whitespace(str(raw.get("image") or ""))
        return self.TrafilaturaResult(
            title=clean_whitespace(str(raw.get("title") or "")),
            published_date=clean_whitespace(str(raw.get("date") or "")),
            author=clean_whitespace(str(raw.get("author") or "")),
            image=urljoin(url, image) if image else "",
            markdown=markdown,
            engine="trafilatura_cleaned",
        )

    def _extract_trafilatura(self, *, raw_html: str, url: str) -> TrafilaturaResult:
        """Standard trafilatura extraction."""
        raw = self._trafilatura_extract(raw_html, url)
        if raw is None:
            raise ValueError("trafilatura bare_extraction returned invalid payload")

        markdown = str(raw.get("text") or raw.get("raw_text") or "").strip()
        if not markdown:
            raise ValueError("trafilatura bare_extraction returned empty markdown")

        image = clean_whitespace(str(raw.get("image") or ""))
        return self.TrafilaturaResult(
            title=clean_whitespace(str(raw.get("title") or "")),
            published_date=clean_whitespace(str(raw.get("date") or "")),
            author=clean_whitespace(str(raw.get("author") or "")),
            image=urljoin(url, image) if image else "",
            markdown=markdown,
            engine="trafilatura",
        )

    def _trafilatura_extract(self, html: str, url: str) -> dict[str, Any] | None:
        """Execute trafilatura bare_extraction and return payload dict."""
        try:
            raw = trafilatura.bare_extraction(
                html,
                url=url,
                include_comments=False,
                output_format="python",
                favor_precision=False,
                favor_recall=True,
                include_tables=True,
                include_images=False,
                include_formatting=True,
                include_links=True,
                deduplicate=False,
                with_metadata=True,
            )
        except Exception:  # trafilatura can raise various internal exceptions
            return None

        if isinstance(raw, dict):
            return raw
        if raw is not None and callable(getattr(raw, "as_dict", None)):
            return cast("dict[str, Any]", raw.as_dict())
        return None

    def _extract_dom_fallback(
        self, *, snapshot: Snapshot, url: str, preserve_html_tags: bool
    ) -> TrafilaturaResult | None:
        """Fallback: extract from semantic HTML regions or content containers.

        Unified method that tries:
        1. Semantic body fragments from snapshot
        2. Full body HTML as fallback
        3. Cleaned content container fragments
        """
        # Collect all candidate fragments
        fragments = [f for f in snapshot.semantic_html.get("body", []) if f]

        # Full body HTML as fallback
        if snapshot.tree.body is not None:
            body_html = str(snapshot.tree.body.html or "")
            if body_html:
                fragments.append(body_html)

        if not fragments:
            return None

        return self._select_best_fragment(
            fragments=fragments,
            url=url,
            preserve_html_tags=preserve_html_tags,
            engine_name="dom_fallback",
            clean_fragments=True,
        )

    def _select_best_fragment(
        self,
        *,
        fragments: list[str],
        url: str,
        preserve_html_tags: bool,
        engine_name: str,
        clean_fragments: bool = False,
    ) -> TrafilaturaResult | None:
        """Select the best fragment from candidates by scoring.

        Args:
            fragments: List of HTML fragments to evaluate
            url: Base URL for resolving relative URLs
            preserve_html_tags: Whether to preserve HTML tags
            engine_name: Engine name for the result
            clean_fragments: Whether to clean script/style tags from fragments
        """
        best_markdown = ""
        best_score = 0.0

        for fragment in fragments:
            if not fragment:
                continue

            # Optionally clean the fragment
            html_to_render = fragment
            if clean_fragments:
                tree = HTMLParser(fragment)
                for selector in ("script", "style", "noscript", "svg"):
                    for node in tree.css(selector):
                        node.decompose()
                html_to_render = str(tree.body.html) if tree.body else ""
                if not html_to_render:
                    continue

            markdown, _ = self._render_fragment(
                fragment_html=html_to_render,
                base_url=url,
                preserve_html_tags=preserve_html_tags,
            )
            if not markdown:
                continue

            score = self._content_score(markdown_to_text(markdown))

            if score > best_score:
                best_markdown = markdown
                best_score = score

        if best_score <= 0.0 or not best_markdown:
            return None

        return self.TrafilaturaResult(markdown=best_markdown, engine=engine_name)

    def _extract_script_fallback(
        self, *, raw_html: str, url: str, preserve_html_tags: bool
    ) -> TrafilaturaResult | None:
        """Fallback: extract content from script tags (for SPAs)."""
        best_markdown = ""
        best_score = 0.0

        for candidate in self._script_candidates(raw_html):
            markdown = self._script_candidate_markdown(
                candidate=candidate,
                url=url,
                preserve_html_tags=preserve_html_tags,
            )
            if not markdown:
                continue
            score = self._content_score(markdown_to_text(markdown))
            if score > best_score:
                best_markdown = markdown
                best_score = score

        if best_score <= 0.0:
            return None
        return self.TrafilaturaResult(markdown=best_markdown, engine="script_fallback")

    # --------------------------------------------------------------------------
    # Script Content Extraction (SPA Support)
    # --------------------------------------------------------------------------

    def _script_candidates(self, raw_html: str) -> list[str]:
        """Extract potential content strings from script tags."""
        out: list[str] = []
        seen: set[str] = set()

        for script in re.findall(
            r"<script[^>]*>(.*?)</script>", raw_html, flags=re.DOTALL | re.IGNORECASE
        ):
            string_literals = [
                match.group("body")
                for match in re.finditer(
                    r'"(?P<body>(?:\\.|[^"\\]){120,})"', script, flags=re.DOTALL
                )
            ]
            string_literals.extend(
                match.group("body")
                for match in re.finditer(
                    r"'(?P<body>(?:\\.|[^'\\]){120,})'", script, flags=re.DOTALL
                )
            )

            for literal in string_literals:
                decoded = self._decode_js_string(literal)
                if not decoded:
                    continue
                if decoded.startswith(("{", "[")):
                    out.extend(self._json_script_candidates(decoded))
                    continue
                text = clean_whitespace(decoded)
                if text and text not in seen:
                    seen.add(text)
                    out.append(text)

        return [item for item in out if self._content_score(item) > 0.0]

    def _json_script_candidates(self, payload_text: str) -> list[str]:
        """Extract content strings from JSON within script tags."""
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return []

        out: list[str] = []
        seen: set[str] = set()

        def visit(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    visit(child_value, str(child_key).lower())
                return
            if isinstance(value, list):
                for child in value:
                    visit(child, key)
                return
            if not isinstance(value, str):
                return

            text = clean_whitespace(self._decode_js_string(value))
            if not text or text in seen:
                return
            if self._content_score(text, key=key) <= 0.0:
                return
            seen.add(text)
            out.append(text)

        visit(payload)
        return out

    def _script_candidate_markdown(
        self, *, candidate: str, url: str, preserve_html_tags: bool
    ) -> str:
        """Convert script candidate to markdown."""
        normalized = clean_whitespace(candidate)
        if not normalized:
            return ""

        if "<" in normalized and ">" in normalized:
            markdown, _ = self._render_fragment(
                fragment_html=normalized,
                base_url=url,
                preserve_html_tags=preserve_html_tags,
            )
            return markdown

        return normalized

    def _decode_js_string(self, value: str) -> str:
        """Decode JavaScript string escapes."""
        decoded = value
        for _ in range(2):
            try:
                next_value = bytes(decoded, "utf-8").decode("unicode_escape")
            except (UnicodeDecodeError, UnicodeEncodeError):
                break
            if next_value == decoded:
                break
            decoded = next_value
        return html_lib.unescape(decoded)

    # --------------------------------------------------------------------------
    # Content Scoring
    # --------------------------------------------------------------------------

    def _content_score(self, text: str, *, key: str = "") -> float:
        """Score text content quality. Higher = better content candidate."""
        normalized = clean_whitespace(text)
        if len(normalized) < self._MIN_CONTENT_SCORE_CHARS:
            return 0.0

        lowered = normalized.casefold()
        if lowered.startswith(
            ("http://", "https://", "window.", "function ", "return ")
        ):
            return 0.0
        if "data:image/" in lowered:
            return 0.0

        url_hits = lowered.count("http://") + lowered.count("https://")
        punctuation = sum(1 for ch in normalized if ch in ",.!?;:，。！？；：")
        spaces = normalized.count(" ")

        if url_hits > 0 and spaces < 8 and punctuation < 12:
            return 0.0
        if len(set(normalized)) < 12:
            return 0.0

        score = float(len(normalized))
        score += punctuation * 6.0
        score += spaces * 1.5
        score -= url_hits * 80.0

        # Bonus for content-related keys
        if any(
            token in key
            for token in (
                "summary",
                "content",
                "markdown",
                "readme",
                "body",
                "description",
                "detail",
                "introduction",
                "overview",
            )
        ):
            score += 240.0

        # Penalty for non-content keys
        if any(
            token in key
            for token in ("license", "sha", "file", "avatar", "url", "path", "image")
        ):
            score -= 120.0

        return max(0.0, score)

    # --------------------------------------------------------------------------
    # Metadata Extraction
    # --------------------------------------------------------------------------

    def _fill_missing_metadata(
        self, *, extracted: TrafilaturaResult, snapshot: Snapshot, url: str
    ) -> TrafilaturaResult:
        """Fill missing metadata from enriched sources."""
        enriched = self._extract_enriched_metadata(tree=snapshot.tree, url=url)

        title = (
            extracted.title
            or enriched.title
            or self._meta_content(
                tree=snapshot.tree,
                selectors=("meta[property='og:title']", "meta[name='title']", "title"),
                attr="content",
            )
        )
        if not title and snapshot.tree.css_first("title") is not None:
            title = self._node_text(snapshot.tree.css_first("title"))

        published_date = (
            extracted.published_date
            or enriched.published_date
            or self._meta_content(
                tree=snapshot.tree,
                selectors=(
                    "meta[property='article:published_time']",
                    "meta[name='article:published_time']",
                    "meta[name='date']",
                    "meta[property='datePublished']",
                ),
                attr="content",
            )
        )

        author = (
            extracted.author
            or enriched.author
            or self._meta_content(
                tree=snapshot.tree,
                selectors=(
                    "meta[name='author']",
                    "meta[property='article:author']",
                    "meta[name='twitter:creator']",
                ),
                attr="content",
            )
        )

        image = (
            extracted.image
            or enriched.image
            or self._meta_content(
                tree=snapshot.tree,
                selectors=(
                    "meta[property='og:image']",
                    "meta[name='twitter:image']",
                    "meta[name='twitter:image:src']",
                ),
                attr="content",
            )
        )

        description = (
            extracted.description
            or enriched.description
            or self._meta_content(
                tree=snapshot.tree,
                selectors=(
                    "meta[property='og:description']",
                    "meta[name='description']",
                    "meta[name='twitter:description']",
                ),
                attr="content",
            )
        )

        return self.TrafilaturaResult(
            title=clean_whitespace(title),
            published_date=clean_whitespace(published_date),
            author=clean_whitespace(author),
            image=urljoin(url, image) if image else "",
            markdown=extracted.markdown,
            engine=extracted.engine,
            description=clean_whitespace(description),
            site_name=enriched.site_name,
            locale=enriched.locale,
        )

    def _extract_enriched_metadata(self, *, tree: HTMLParser, url: str) -> EnrichedMeta:
        """Extract metadata from JSON-LD, OpenGraph, Twitter Cards, and meta tags."""
        meta = self.EnrichedMeta()
        self._extract_jsonld_metadata(tree, meta)
        self._extract_opengraph_metadata(tree, meta, url)
        self._extract_twitter_metadata(tree, meta)
        self._extract_standard_meta(tree, meta)
        return meta

    def _extract_jsonld_metadata(self, tree: HTMLParser, meta: EnrichedMeta) -> None:
        """Extract metadata from JSON-LD structured data."""
        for script in tree.css('script[type="application/ld+json"]'):
            try:
                text = script.text()
                if not text:
                    continue
                data = json.loads(text)
            except (json.JSONDecodeError, Exception):  # noqa: BLE001, S112
                continue

            # Handle @graph arrays
            items: list[dict[str, Any]] = []
            if isinstance(data, dict):
                if "@graph" in data:
                    items = [item for item in data["@graph"] if isinstance(item, dict)]
                else:
                    items = [data]
            elif isinstance(data, list):
                items = [item for item in data if isinstance(item, dict)]

            for item in items:
                type_name = str(item.get("@type", "")).lower()

                # Article, BlogPosting, NewsArticle, etc.
                if any(
                    t in type_name
                    for t in (
                        "article",
                        "blog",
                        "news",
                        "post",
                        "documentation",
                        "webpage",
                        "page",
                    )
                ):
                    if not meta.title and "headline" in item:
                        meta.title = clean_whitespace(str(item["headline"]))
                    if not meta.description and "description" in item:
                        meta.description = clean_whitespace(str(item["description"]))
                    if not meta.author:
                        author = item.get("author")
                        if isinstance(author, dict):
                            meta.author = clean_whitespace(str(author.get("name", "")))
                        elif isinstance(author, str):
                            meta.author = clean_whitespace(author)
                    if not meta.published_date and "datePublished" in item:
                        meta.published_date = clean_whitespace(
                            str(item["datePublished"])
                        )
                    if not meta.modified_date and "dateModified" in item:
                        meta.modified_date = clean_whitespace(str(item["dateModified"]))
                    if not meta.image:
                        image = item.get("image")
                        if isinstance(image, str):
                            meta.image = image
                        elif isinstance(image, dict) and "url" in image:
                            meta.image = str(image["url"])
                        elif isinstance(image, list) and image:
                            first = image[0]
                            if isinstance(first, str):
                                meta.image = first
                            elif isinstance(first, dict) and "url" in first:
                                meta.image = str(first["url"])

                # Organization, WebSite for site_name
                if "organization" in type_name or "website" in type_name:
                    if not meta.site_name and "name" in item:
                        meta.site_name = clean_whitespace(str(item["name"]))

                # BreadcrumbList for keywords/context
                if "breadcrumb" in type_name:
                    item_list = item.get("itemListElement", [])
                    if isinstance(item_list, list):
                        for elem in item_list:
                            if isinstance(elem, dict) and "name" in elem:
                                name = clean_whitespace(str(elem["name"]))
                                if name and name not in meta.keywords:
                                    meta.keywords.append(name)

    def _extract_opengraph_metadata(
        self, tree: HTMLParser, meta: EnrichedMeta, url: str
    ) -> None:
        """Extract metadata from OpenGraph tags."""
        og_fields = {
            "og:title": "title",
            "og:description": "description",
            "og:site_name": "site_name",
            "og:locale": "locale",
            "article:author": "author",
            "article:published_time": "published_date",
            "article:modified_time": "modified_date",
        }

        for prop, field in og_fields.items():
            if getattr(meta, field):
                continue
            node = tree.css_first(f"meta[property='{prop}']")
            if node:
                value = clean_whitespace(str(node.attributes.get("content", "")))
                if value:
                    setattr(meta, field, value)

        # og:image can be complex (array or single)
        if not meta.image:
            for prop in ("og:image", "og:image:url"):
                node = tree.css_first(f"meta[property='{prop}']")
                if node:
                    value = clean_whitespace(str(node.attributes.get("content", "")))
                    if value:
                        meta.image = value
                        break

    def _extract_twitter_metadata(self, tree: HTMLParser, meta: EnrichedMeta) -> None:
        """Extract metadata from Twitter Card tags."""
        twitter_fields = {
            "twitter:title": "title",
            "twitter:description": "description",
            "twitter:creator": "author",
            "twitter:site": "site_name",
        }

        for name, field in twitter_fields.items():
            if getattr(meta, field):
                continue
            node = tree.css_first(f"meta[name='{name}']")
            if node:
                value = clean_whitespace(str(node.attributes.get("content", "")))
                if value:
                    if field == "author" and value.startswith("@"):
                        value = value[1:]
                    setattr(meta, field, value)

        if not meta.image:
            for name in ("twitter:image", "twitter:image:src"):
                node = tree.css_first(f"meta[name='{name}']")
                if node:
                    value = clean_whitespace(str(node.attributes.get("content", "")))
                    if value:
                        meta.image = value
                        break

    def _extract_standard_meta(self, tree: HTMLParser, meta: EnrichedMeta) -> None:
        """Extract metadata from standard meta tags."""
        if not meta.description:
            node = tree.css_first("meta[name='description']")
            if node:
                meta.description = clean_whitespace(
                    str(node.attributes.get("content", ""))
                )

        if not meta.author:
            node = tree.css_first("meta[name='author']")
            if node:
                meta.author = clean_whitespace(str(node.attributes.get("content", "")))

        if not meta.keywords:
            node = tree.css_first("meta[name='keywords']")
            if node:
                keywords_str = clean_whitespace(str(node.attributes.get("content", "")))
                if keywords_str:
                    meta.keywords = [
                        kw.strip() for kw in keywords_str.split(",") if kw.strip()
                    ]

    def _meta_content(
        self, *, tree: HTMLParser, selectors: tuple[str, ...], attr: str
    ) -> str:
        """Extract content from first matching meta selector."""
        for selector in selectors:
            node = tree.css_first(selector)
            if node is None:
                continue
            if attr == "content":
                value = clean_whitespace(
                    str(node.attributes.get("content", "")).strip()
                )
            else:
                value = self._node_text(node)
            if value:
                return value
        return ""

    # --------------------------------------------------------------------------
    # Document Rendering
    # --------------------------------------------------------------------------

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
        """Render final document from extracted content."""
        blocks: list[str] = []
        secondary_markdown = ""
        stats_items: list[dict[str, int | float | str | bool]] = []

        for tag in cls._SEMANTIC_ORDER:
            if tag not in selected_tags:
                continue

            if tag == "body":
                content = extracted.markdown
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
                content = "\n\n".join(md for md, _ in rendered if md).strip()
                stats = cls._merge_render_stats([s for _, s in rendered])

            if not content:
                continue

            if tag == "sidebar":
                secondary_markdown = content
            stats_items.append(stats)

            blocks.append(cls._wrap_block(tag, content, preserve_html_tags))

        return cls.RenderBundle(
            markdown="\n\n".join(blocks).strip(),
            secondary_markdown=secondary_markdown,
            stats=cls._merge_render_stats(stats_items),
        )

    @classmethod
    def _render_fragment(
        cls, *, fragment_html: str, base_url: str, preserve_html_tags: bool
    ) -> tuple[str, dict[str, int | float | str | bool]]:
        """Render HTML fragment to markdown."""
        tree = HTMLParser(fragment_html)

        # Resolve relative URLs
        for node in tree.css(
            "[href], [src], [poster], [data-src], [srcset], [data-srcset]"
        ):
            for attr in ("href", "src", "poster", "data-src"):
                raw = str(node.attributes.get(attr, "")).strip()
                if raw:
                    node.attributes[attr] = urljoin(base_url, raw)
            for attr in ("srcset", "data-srcset"):
                raw = str(node.attributes.get(attr, "")).strip()
                if not raw:
                    continue

                # Parse srcset: "url1 1x, url2 2x" or "url1 100w, url2 200w"
                parsed_items = []
                for candidate in raw.split(","):
                    item = candidate.strip()
                    if not item:
                        continue

                    # Split into URL and optional descriptor
                    parts = item.split(None, 1)  # Split on any whitespace, max 1 split
                    url = parts[0]
                    descriptor = parts[1] if len(parts) > 1 else ""

                    resolved_url = urljoin(base_url, url)
                    if descriptor:
                        parsed_items.append(f"{resolved_url} {descriptor}")
                    else:
                        parsed_items.append(resolved_url)

                node.attributes[attr] = ", ".join(parsed_items)

        # Get prepared HTML from body or full tree
        prepared = (
            str(tree.body.html or "")
            if tree.body is not None
            else str(tree.html or fragment_html)
        )

        # Determine conversion options
        if preserve_html_tags:
            # Reuse tree to collect tags (avoid creating another HTMLParser)
            tag_source = tree.body if tree.body is not None else tree
            preserve_tags = {
                str(node.tag).lower()
                for node in tag_source.css("*")
                if str(node.tag or "").strip()
            }
            options = replace(cls._CONVERSION_OPTIONS, preserve_tags=preserve_tags)
        else:
            options = cls._CONVERSION_OPTIONS

        result = convert_with_tables(
            prepared,
            options=options,
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
            fragment_html=prepared, markdown=markdown, result=result
        )

    @classmethod
    def _wrap_block(
        cls, tag: ExtractContentTag, content: str, preserve_html_tags: bool
    ) -> str:
        """Wrap content block with appropriate formatting."""
        if preserve_html_tags:
            return f'<section data-serpsage-tag="{tag}">\n{content}\n</section>'
        if tag in ("body", "metadata"):
            return content
        if tag == "sidebar":
            return f"## Secondary Content\n\n{content}"
        return f"## {tag.capitalize()}\n\n{content}"

    # --------------------------------------------------------------------------
    # Statistics & Warnings
    # --------------------------------------------------------------------------

    def _stats(
        self,
        *,
        markdown: str,
        body_markdown: str,
        secondary_markdown: str,
        render_stats: dict[str, int | float | str | bool],
        primary_engine: str,
        detail: Literal["concise", "standard", "full"],
        include_secondary: bool,
        selected_tags: set[ExtractContentTag],
    ) -> dict[str, int | float | str | bool]:
        """Generate extraction statistics."""
        text_value = markdown_to_text(markdown)
        body_text = markdown_to_text(body_markdown)
        secondary_text = markdown_to_text(secondary_markdown)

        use_fragment_renderer = any(
            tag in selected_tags
            for tag in ("header", "navigation", "banner", "sidebar", "footer")
        )
        backend = (
            f"{primary_engine}+html_to_markdown"
            if use_fragment_renderer
            else primary_engine
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
            "renderer_fallback_used": primary_engine != "trafilatura",
            "renderer_text_recall_ratio": (
                float(render_stats.get("renderer_text_recall_ratio", 1.0))
                if use_fragment_renderer
                else 1.0
            ),
            **{key: int(render_stats.get(key, 0)) for key in self._COUNT_KEYS},
        }

    def _warnings(
        self, *, primary_chars: int, text_chars: int, include_secondary: bool
    ) -> list[str]:
        """Generate extraction warnings."""
        warnings: list[str] = []
        if primary_chars < self._profile.min_primary_chars:
            warnings.append("primary content is short")
        if (
            include_secondary
            and text_chars < self._profile.min_total_chars_with_secondary
        ):
            warnings.append("extracted text is short with secondary content")
        return warnings

    def _assert_nonempty(self, *, markdown: str) -> None:
        """Assert that extracted content is non-empty."""
        text_chars = len(clean_whitespace(markdown_to_text(markdown)))
        if not markdown.strip() or text_chars < max(1, self._profile.min_text_chars):
            raise ValueError("filtered content empty after cleanup")

    # --------------------------------------------------------------------------
    # Link & Image Extraction
    # --------------------------------------------------------------------------

    def _links(
        self, *, snapshot: Snapshot, base_url: str, include_secondary: bool
    ) -> list[ExtractRef]:
        """Extract links from HTML document."""
        base_norm = (
            self._normalize_url(
                url=base_url, base_url=base_url, keep_hash=self._profile.link_keep_hash
            )
            or base_url
        )
        base_netloc = urlparse(base_norm).netloc.lower()
        max_count = self._profile.link_max_count

        collected: list[tuple[int, int, ExtractRef]] = []
        seen: set[tuple[str, str]] = set()

        for position, anchor in enumerate(snapshot.tree.css("a"), start=1):
            href = str(anchor.attributes.get("href", "")).strip()
            text = self._node_text(anchor)
            if not href or not text:
                continue

            normalized = self._normalize_url(
                url=href, base_url=base_url, keep_hash=self._profile.link_keep_hash
            )
            if not normalized:
                continue

            section = self._section(anchor=anchor, snapshot=snapshot)
            if section == "secondary" and not include_secondary:
                continue

            # Dedupe by (url, text)
            dedupe_key = (normalized, text.casefold())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            parsed = urlparse(normalized)
            rel = str(anchor.attributes.get("rel", "")).strip().lower()
            ref = ExtractRef(
                url=normalized,
                text=text,
                zone=cast("Literal['primary', 'secondary']", section),
                internal=(parsed.netloc.lower() == base_netloc)
                if parsed.netloc
                else True,
                nofollow=("nofollow" in rel.split()),
                same_page=(
                    self._strip_fragment(normalized) == self._strip_fragment(base_norm)
                ),
                source=self._source_hint(anchor),
                position=position,
            )
            score = self._link_importance_score(ref=ref)
            collected.append((score, position, ref))

        # Sort by importance (desc), then by position (asc)
        collected.sort(key=lambda item: (-item[0], item[1]))
        return [ref for _, _, ref in collected[:max_count]]

    @staticmethod
    def _link_importance_score(*, ref: ExtractRef) -> int:
        """Calculate importance score for a link. Higher = more important."""
        return 100 if ref.zone == "primary" else 0

    def _images(
        self, *, snapshot: Snapshot, base_url: str, include_secondary: bool
    ) -> list[ExtractRef]:
        """Extract images from HTML document."""
        base_norm = (
            self._normalize_url(
                url=base_url, base_url=base_url, keep_hash=self._profile.link_keep_hash
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

    # --------------------------------------------------------------------------
    # Node Utilities
    # --------------------------------------------------------------------------

    @classmethod
    def _node_ident(cls, node: Node) -> str:
        """Get node identifier (id + class) for pattern matching."""
        return " ".join(
            [
                str(node.attributes.get("id", "")).strip(),
                str(node.attributes.get("class", "")).strip(),
            ]
        ).strip()

    @classmethod
    def _unique_html(cls, nodes: list[Node]) -> list[str]:
        """Get unique HTML strings from nodes, deduplicating by path."""
        seen: set[str] = set()
        items: list[str] = []
        for node in cls._dedupe_nodes(nodes):
            html_value = str(node.html or "").strip()
            if html_value and html_value not in seen:
                seen.add(html_value)
                items.append(html_value)
        return items

    @classmethod
    def _dedupe_nodes(cls, nodes: list[Node]) -> list[Node]:
        """Deduplicate nodes by path, removing nested duplicates."""
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
        """Determine if node is in primary or secondary content."""
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
    def _is_secondary(cls, node: Node) -> bool:
        """Check if node is a secondary content region."""
        ident = cls._node_ident(node)
        # Normalize style: lowercase and remove ALL whitespace (spaces, tabs, newlines)
        style = re.sub(
            r"\s+", "", str(node.attributes.get("style", "")).strip().lower()
        )
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
        """Check if node is a banner region."""
        role = str(node.attributes.get("role", "")).strip().lower()
        ident = cls._node_ident(node)
        return role == "banner" or bool(ident and cls._BANNER_RE.search(ident))

    @classmethod
    def _favicon(cls, *, tree: HTMLParser, base_url: str) -> str:
        """Extract favicon URL from HTML."""
        for node in tree.css("link[rel]"):
            rel = str(node.attributes.get("rel", "")).strip().lower()
            href = str(node.attributes.get("href", "")).strip()
            if "icon" in rel and href:
                normalized = cls._normalize_url(
                    url=href, base_url=base_url, keep_hash=True
                )
                if normalized:
                    return normalized
        return ""

    @classmethod
    def _source_hint(cls, node: Node) -> str:
        """Get source hint from node or ancestors."""
        current: Node | None = node
        for _ in range(4):
            if current is None:
                break
            ident = cls._node_ident(current)
            if ident:
                return ident[:120]
            current = current.parent
        return "unknown"

    @classmethod
    def _image_candidates(cls, image: Node) -> list[str]:
        """Extract image URL candidates from node."""
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
    def _text_len(cls, node: Node | None) -> int:
        """Get text length of node."""
        return len(cls._node_text(node))

    @classmethod
    def _node_text(cls, node: Node | None) -> str:
        """Get cleaned text content of node."""
        return (
            ""
            if node is None
            else clean_whitespace(node.text(separator=" ", strip=True))
        )

    @classmethod
    def _is_descendant(cls, *, parent: Node, child: Node) -> bool:
        """Check if child is a descendant of parent."""
        current: Node | None = child
        while current is not None:
            if current is parent:
                return True
            current = current.parent
        return False

    # --------------------------------------------------------------------------
    # Node Path Utilities (for content region detection)
    # --------------------------------------------------------------------------

    @classmethod
    def _node_path(cls, node: Node | None) -> tuple[int, ...] | None:
        """Get path from root to node as tuple of sibling indices."""
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
        """Check if path starts with prefix."""
        return len(path) >= len(prefix) and path[: len(prefix)] == prefix

    @classmethod
    def _node_id(cls, node: Node) -> int:
        """Get unique node ID."""
        value = node.mem_id
        return int(value()) if callable(value) else int(value)

    # --------------------------------------------------------------------------
    # URL Utilities
    # --------------------------------------------------------------------------

    @classmethod
    def _normalize_url(cls, *, url: str, base_url: str, keep_hash: bool) -> str | None:
        """Normalize URL, removing tracking parameters."""
        raw = (url or "").strip()
        if not raw or raw.lower().startswith(
            ("javascript:", "mailto:", "tel:", "data:")
        ):
            return None

        try:
            parsed = urlparse(urljoin(base_url, raw))
        except ValueError:
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
        """Remove fragment from URL."""
        parsed = urlparse(url)
        return urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, "")
        )

    # --------------------------------------------------------------------------
    # Render Statistics
    # --------------------------------------------------------------------------

    @classmethod
    def _empty_render_stats(cls) -> dict[str, int | float | str | bool]:
        """Return empty render statistics."""
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
        """Merge multiple render statistics."""
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
        cls, *, fragment_html: str, markdown: str, result: TableExtractionResult
    ) -> dict[str, int | float | str | bool]:
        """Calculate render statistics for fragment."""
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
    def _has_pre_ancestor(cls, node: Node) -> bool:
        """Check if node has a <pre> ancestor."""
        current = node.parent
        while current is not None:
            if str(current.tag or "").lower() == "pre":
                return True
            current = current.parent
        return False

    @classmethod
    def _block_count(cls, fragment_html: str) -> int:
        """Count block-level elements in fragment."""
        return sum(
            1
            for node in HTMLParser(fragment_html).css(",".join(sorted(cls._BLOCK_TAGS)))
            if cls._node_text(node)
        )


__all__ = ["HtmlExtractor", "HtmlExtractorConfig"]
