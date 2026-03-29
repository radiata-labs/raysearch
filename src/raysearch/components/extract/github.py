"""GitHub repository page extractor.

Extracts structured content from GitHub repository pages including:
- Repository metadata (stars, forks, language, topics)
- README content (cleanly extracted)
- Sidebar with repo info
- Minimal navigation noise

GitHub pages have consistent structure, making them ideal for specialized parsing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from typing_extensions import override
from urllib.parse import urljoin, urlparse

import trafilatura
from html_to_markdown import (
    ConversionOptions,
    MetadataConfig,
    PreprocessingOptions,
    convert_with_tables,
)
from selectolax.parser import HTMLParser

from raysearch.components.extract.base import (
    ExtractConfigBase,
    SpecializedExtractorBase,
)
from raysearch.models.components.extract import (
    ExtractContent,
    ExtractContentTag,
    ExtractedDocument,
    ExtractMeta,
    ExtractRef,
    ExtractRefs,
    ExtractSpec,
    ExtractTrace,
)
from raysearch.utils import clean_whitespace


class GitHubExtractorConfig(ExtractConfigBase):
    __setting_family__ = "extract"
    __setting_name__ = "github"


@dataclass(slots=True)
class GitHubRepoMeta:
    """Extracted GitHub repository metadata."""

    owner: str = ""
    repo: str = ""
    full_name: str = ""
    description: str = ""
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    language: str = ""
    topics: list[str] = field(default_factory=list)
    license_name: str = ""
    homepage_url: str = ""
    is_fork: bool = False
    is_archived: bool = False


@dataclass(slots=True)
class GitHubSnapshot:
    """Snapshot of parsed GitHub page."""

    tree: HTMLParser
    repo_meta: GitHubRepoMeta
    semantic_html: dict[ExtractContentTag, list[str]]
    favicon: str


class GitHubExtractor(SpecializedExtractorBase[GitHubExtractorConfig]):
    """Specialized extractor for GitHub repository pages."""

    _GITHUB_HOSTS: ClassVar[set[str]] = {
        "github.com",
        "www.github.com",
    }

    _REPO_URL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^/([^/]+)/([^/]+?)(?:/|$|\.git$)"
    )

    _DETAIL_DEFAULT_TAGS: ClassVar[dict[str, tuple[ExtractContentTag, ...]]] = {
        "concise": ("body",),
        "standard": ("header", "body", "sidebar"),
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

    @classmethod
    @override
    def can_handle(
        cls,
        *,
        url: str,
        content_type: str | None,
        crawl_backend: str = "curl_cffi",
        content_kind: Literal[
            "html", "pdf", "text", "markdown", "json", "binary", "unknown"
        ] = "unknown",
        content: bytes | None = None,
    ) -> bool:
        # Only handle HTML content
        if content_kind not in ("html", "unknown"):
            return False
        if content_type and "html" not in content_type.lower():
            return False

        try:
            parsed = urlparse(url)
        except Exception:  # noqa: BLE001
            return False

        host = (parsed.netloc or "").lower()
        if host not in cls._GITHUB_HOSTS:
            return False

        path = parsed.path.rstrip("/")
        if not path or path == "/":
            return False

        match = cls._REPO_URL_PATTERN.match(path)
        if not match:
            return False

        skip_paths = {
            "/explore",
            "/notifications",
            "/pulls",
            "/issues",
            "/marketplace",
            "/search",
            "/settings",
            "/organizations",
            "/new",
            "/login",
            "/signup",
            "/pricing",
            "/features",
            "/security",
            "/codespaces",
            "/sponsors",
            "/trending",
            "/collections",
            "/topics",
        }
        return path.lower() not in skip_paths

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
        raw_html = content.decode("utf-8", errors="replace")
        snapshot = self._snapshot(raw_html=raw_html, base_url=url)

        options = content_options or ExtractSpec()
        selected_tags = self._resolve_tags(options)

        # Extract primary content (README + page content via trafilatura)
        primary_result = self._extract_primary(
            raw_html=raw_html,
            snapshot=snapshot,
            url=url,
            preserve_html_tags=bool(options.keep_html),
        )

        # Render document with selected sections
        rendered = self._render_document(
            snapshot=snapshot,
            primary_result=primary_result,
            selected_tags=selected_tags,
            url=url,
            preserve_html_tags=bool(options.keep_html),
        )

        # Build stats
        stats = self._build_stats(
            repo_meta=snapshot.repo_meta,
            markdown=rendered.markdown,
            detail=options.detail,
            selected_tags=selected_tags,
            engine=primary_result.engine,
        )

        # Extract links and images if requested
        links: list[ExtractRef] = []
        images: list[ExtractRef] = []
        if collect_links:
            links = self._extract_links(snapshot=snapshot, url=url)
        if collect_images:
            images = self._extract_images(snapshot=snapshot, url=url)

        engine_val = str(stats.get("engine_chain", "github"))
        doc = ExtractedDocument(
            content=ExtractContent(markdown=rendered.markdown),
            meta=ExtractMeta(
                title=primary_result.title or snapshot.repo_meta.full_name or url,
                author=snapshot.repo_meta.owner,
                image=primary_result.image,
                favicon=snapshot.favicon,
            ),
            refs=ExtractRefs(links=links, images=images),
            trace=ExtractTrace(
                kind="html",
                engine=engine_val,
                stats=stats,
            ),
        )

        return self._finalize_content(doc=doc, content_options=options)

    def _resolve_tags(self, options: ExtractSpec) -> set[ExtractContentTag]:
        if options.sections:
            selected = set(options.sections)
        else:
            selected = set(self._DETAIL_DEFAULT_TAGS.get(options.detail, ("body",)))
        return selected

    def _snapshot(self, *, raw_html: str, base_url: str) -> GitHubSnapshot:
        tree = HTMLParser(raw_html)
        repo_meta = self._extract_repo_meta(tree=tree, url=base_url)

        # Extract all semantic HTML sections
        semantic_html: dict[ExtractContentTag, list[str]] = {
            "header": self._extract_header_html(tree),
            "navigation": self._extract_navigation_html(tree),
            "banner": self._extract_banner_html(tree),
            "body": self._extract_body_html(tree),
            "sidebar": self._extract_sidebar_html(tree),
            "footer": self._extract_footer_html(tree),
        }

        return GitHubSnapshot(
            tree=tree,
            repo_meta=repo_meta,
            semantic_html=semantic_html,
            favicon="https://github.com/fluidicon.png",
        )

    def _extract_repo_meta(self, *, tree: HTMLParser, url: str) -> GitHubRepoMeta:
        meta = GitHubRepoMeta()

        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        match = self._REPO_URL_PATTERN.match(path)
        if match:
            meta.owner = match.group(1)
            meta.repo = match.group(2)
            meta.full_name = f"{meta.owner}/{meta.repo}"

        self._extract_from_meta_tags(tree, meta)
        self._extract_from_structured_data(tree, meta)
        self._extract_from_page_elements(tree, meta)

        return meta

    def _extract_from_meta_tags(self, tree: HTMLParser, meta: GitHubRepoMeta) -> None:
        og_title = self._get_meta_content(tree, "og:title", use_property=True)
        if og_title:
            if ":" in og_title:
                title_part, desc_part = og_title.split(":", 1)
                if not meta.description:
                    meta.description = clean_whitespace(desc_part)
            else:
                title_part = og_title
            if "/" in title_part and not meta.full_name:
                parts = title_part.strip().split("/")
                if len(parts) >= 2:
                    meta.owner = parts[0].strip()
                    meta.repo = parts[1].strip()
                    meta.full_name = f"{meta.owner}/{meta.repo}"

        og_desc = self._get_meta_content(tree, "og:description", use_property=True)
        if og_desc and not meta.description:
            meta.description = og_desc

    def _extract_from_structured_data(
        self, tree: HTMLParser, meta: GitHubRepoMeta
    ) -> None:
        for script in tree.css('script[type="application/ld+json"]'):
            try:
                text = script.text()
                if not text:
                    continue
                data = json.loads(text)
            except (json.JSONDecodeError, Exception):  # noqa: BLE001, S112
                continue

            if not isinstance(data, dict):
                continue

            if "@graph" in data:
                for item in data["@graph"]:
                    if isinstance(item, dict):
                        self._apply_jsonld_item(item, meta)
            else:
                self._apply_jsonld_item(data, meta)

    def _apply_jsonld_item(self, data: dict[str, Any], meta: GitHubRepoMeta) -> None:
        if "name" in data and not meta.repo:
            name = str(data["name"])
            if "/" in name:
                parts = name.split("/")
                if len(parts) == 2:
                    meta.owner = parts[0].strip()
                    meta.repo = parts[1].strip()
                    meta.full_name = name.strip()
            else:
                meta.repo = name

        if "description" in data and not meta.description:
            meta.description = clean_whitespace(str(data["description"]))

        if "author" in data and not meta.owner:
            author = data["author"]
            if isinstance(author, dict):
                meta.owner = str(author.get("name", ""))
            elif isinstance(author, str):
                meta.owner = author

        if "programmingLanguage" in data and not meta.language:
            lang = data["programmingLanguage"]
            if isinstance(lang, list) and lang:
                meta.language = str(lang[0])
            elif isinstance(lang, str):
                meta.language = lang

    def _extract_from_page_elements(
        self, tree: HTMLParser, meta: GitHubRepoMeta
    ) -> None:
        if not meta.stars:
            meta.stars = self._extract_count(tree, "stargazers")
        if not meta.forks:
            meta.forks = self._extract_count(tree, "forks")
        if not meta.watchers:
            meta.watchers = self._extract_count(tree, "watchers")
        if not meta.language:
            meta.language = self._extract_language(tree)
        if not meta.topics:
            meta.topics = self._extract_topics(tree)
        if not meta.license_name:
            meta.license_name = self._extract_license(tree)

        meta.is_archived = self._check_archived(tree)
        meta.is_fork = self._check_fork(tree)

    def _extract_count(self, tree: HTMLParser, count_type: str) -> int:
        selectors = [
            f'a[href$="/{count_type}"] span',
            f'a[href*="/{count_type}"] span',
        ]

        for selector in selectors:
            for node in tree.css(selector):
                parent = node.parent
                if parent and parent.attributes.get("href"):
                    href = str(parent.attributes.get("href", ""))
                    if count_type in href:
                        return self._parse_count(node.text(strip=True))

        for elem in tree.css("[aria-label]"):
            aria = str(elem.attributes.get("aria-label", "")).lower()
            if count_type in aria:
                count = self._parse_count(elem.text(strip=True))
                if count > 0:
                    return count

        return 0

    def _parse_count(self, text: str) -> int:
        text = text.strip().lower().replace(",", "").replace(" ", "")
        if not text:
            return 0
        try:
            if text.endswith("k"):
                return int(float(text[:-1]) * 1000)
            if text.endswith("m"):
                return int(float(text[:-1]) * 1000000)
            return int(float(text))
        except (ValueError, TypeError):
            return 0

    def _extract_language(self, tree: HTMLParser) -> str:
        lang_node = tree.css_first('[itemprop="programmingLanguage"]')
        if lang_node:
            return clean_whitespace(lang_node.text())

        for elem in tree.css("[class]"):
            if "language-color" in str(elem.attributes.get("class", "")):
                parent = elem.parent
                if parent:
                    text = parent.text(strip=True)
                    if text:
                        parts = text.split()
                        if parts:
                            return parts[0]
        return ""

    def _extract_topics(self, tree: HTMLParser) -> list[str]:
        topics: list[str] = []
        for node in tree.css("a.topic-tag, a[data-ga-click*='topic']"):
            text = node.text(strip=True)
            if text and text not in topics:
                topics.append(text)
        return topics

    def _extract_license(self, tree: HTMLParser) -> str:
        for elem in tree.css("div, li, span"):
            text = elem.text(strip=True)
            if text.lower().startswith("license"):
                parts = text.split()
                if len(parts) >= 2:
                    license_name = " ".join(parts[1:])
                    if license_name.lower() not in (
                        "license",
                        "view license",
                        "see license",
                    ):
                        return license_name

        for link in tree.css("a"):
            href = str(link.attributes.get("href", ""))
            if "/license" in href.lower() and href.endswith("/license"):
                text = link.text(strip=True)
                if text and text.lower() not in (
                    "license",
                    "view license",
                    "see license",
                    "mit",
                    "apache",
                    "gpl",
                ):
                    return text
                title = link.attributes.get("title")
                if title and str(title).lower() != "license":
                    return str(title)

        return ""

    def _check_archived(self, tree: HTMLParser) -> bool:
        for elem in tree.css("[class]"):
            if "archived" in str(elem.attributes.get("class", "")).lower():
                return True
        return False

    def _check_fork(self, tree: HTMLParser) -> bool:
        for elem in tree.css("span, div"):
            if elem.text(strip=True).lower().startswith("forked from"):
                return True
        return False

    def _get_meta_content(
        self, tree: HTMLParser, name: str, *, use_property: bool = False
    ) -> str:
        selector = (
            f"meta[property='{name}']" if use_property else f"meta[name='{name}']"
        )
        node = tree.css_first(selector)
        return str(node.attributes.get("content", "")).strip() if node else ""

    def _extract_header_html(self, tree: HTMLParser) -> list[str]:
        """Extract repository header HTML (repo name row with star/fork/watch buttons)."""
        fragments: list[str] = []
        # The repo title row is in .d-flex.mb-3.pb-3 within #repository-container-header
        # We need to avoid including the navigation tabs which are in UnderlineNav
        repo_header = tree.css_first("#repository-container-header")
        if repo_header:
            # Get HTML string before any modifications
            header_html = repo_header.html
            if header_html:
                # Parse a copy to avoid modifying the original tree
                header_tree = HTMLParser(header_html)
                repo_header_copy = header_tree.css_first("#repository-container-header")
                if repo_header_copy:
                    # Remove navigation tabs to avoid duplication
                    for nav in repo_header_copy.css("nav.UnderlineNav, .js-repo-nav"):
                        nav.decompose()
                    # Remove the file navigation too
                    for file_nav in repo_header_copy.css(".js-repo-nav-wrapper"):
                        file_nav.decompose()
                    # Clean up noisy elements
                    for svg in repo_header_copy.css("svg"):
                        svg.decompose()
                    for btn in repo_header_copy.css(
                        "button, .js-zeroclipboard, [data-copy-feedback]"
                    ):
                        btn.decompose()
                    # Remove session alerts
                    for alert in repo_header_copy.css(".js-flash-container, .flash"):
                        alert.decompose()
                    # Remove mobile-only duplicate elements (usually in details/summary)
                    for details in repo_header_copy.css(
                        "details, .d-md-none, .hide-md"
                    ):
                        details.decompose()
                    cleaned_html = repo_header_copy.html
                    if cleaned_html:
                        fragments.append(cleaned_html)
        return self._unique_html_fragments(fragments)

    def _extract_navigation_html(self, tree: HTMLParser) -> list[str]:
        """Extract repository navigation HTML (Code, Issues, Pull requests, Actions, etc.)."""
        fragments: list[str] = []
        # GitHub repo nav is in nav.js-repo-nav or UnderlineNav
        repo_nav = tree.css_first("nav.js-repo-nav, nav.UnderlineNav")
        if repo_nav:
            nav_html = repo_nav.html
            if nav_html:
                # Parse a copy to avoid modifying the original tree
                nav_tree = HTMLParser(nav_html)
                # Clean up noisy elements
                for svg in nav_tree.css("svg"):
                    svg.decompose()
                for btn in nav_tree.css(
                    "button, .js-zeroclipboard, [data-copy-feedback]"
                ):
                    btn.decompose()
                # Remove mobile-only duplicate navigation (dropdown menus)
                for dropdown in nav_tree.css(
                    ".UnderlineNav-actions, .js-responsive-underlinenav-overflow"
                ):
                    dropdown.decompose()
                # Remove "Additional navigation options" section
                for details in nav_tree.css("details, details-summary"):
                    details.decompose()
                cleaned_html = nav_tree.body.html if nav_tree.body else ""
                if cleaned_html:
                    fragments.append(cleaned_html)
        return self._unique_html_fragments(fragments)

    def _extract_banner_html(self, tree: HTMLParser) -> list[str]:
        """Extract banner/hero section HTML (archived/disabled warnings only)."""
        fragments: list[str] = []
        # Only extract repo status banners (archived, disabled, etc.)
        # NOT session alerts like "You signed in with another tab"
        for selector in (".status-archived", ".status-disabled", ".repo-status"):
            banner = tree.css_first(selector)
            if banner:
                banner_html = banner.html
                if banner_html:
                    # Parse a copy to clean up
                    banner_tree = HTMLParser(banner_html)
                    for svg in banner_tree.css("svg"):
                        svg.decompose()
                    cleaned_html = banner_tree.body.html if banner_tree.body else ""
                    if cleaned_html:
                        fragments.append(cleaned_html)
        return self._unique_html_fragments(fragments)

    def _extract_body_html(self, tree: HTMLParser) -> list[str]:
        """Extract main body content HTML (entire main element including files table and README)."""
        fragments: list[str] = []

        # Get main element - this contains both the files table and README
        main = tree.css_first("main")
        if main:
            main_html = main.html
            if main_html:
                # Parse a copy to avoid modifying the original tree
                main_tree = HTMLParser(main_html)
                main_copy = main_tree.css_first("main") or main_tree.body
                if main_copy:
                    # Clean up SVGs and noisy elements in the main content
                    for svg in main_copy.css("svg"):
                        svg.decompose()
                    for anchor in main_copy.css("a.anchor"):
                        anchor.decompose()
                    for btn in main_copy.css(
                        "button, .js-zeroclipboard, [data-copy-feedback]"
                    ):
                        btn.decompose()
                    cleaned_html = main_copy.html
                    if cleaned_html:
                        fragments.append(cleaned_html)
                        return self._unique_html_fragments(fragments)

        # Fallback to article (README only)
        for selector in ("article.markdown-body", "div.markdown-body", "article"):
            node = tree.css_first(selector)
            if node:
                node_html = node.html
                if node_html:
                    node_tree = HTMLParser(node_html)
                    self._clean_readme_node(node_tree.body or node_tree)
                    cleaned_html = node_tree.body.html if node_tree.body else ""
                    if cleaned_html:
                        fragments.append(cleaned_html)
                        return self._unique_html_fragments(fragments)

        # Final fallback to body
        if tree.body and tree.body.html:
            fragments.append(tree.body.html)

        return self._unique_html_fragments(fragments)

    def _extract_sidebar_html(self, tree: HTMLParser) -> list[str]:
        """Extract GitHub repo sidebar HTML (About section, topics, releases, etc.)."""
        fragments: list[str] = []
        # GitHub updated their layout - sidebar is now in PageLayout-Pane
        # Try multiple selectors for backwards compatibility
        sidebar = tree.css_first(
            "div.Layout-sidebar, [class*='PageLayout-Pane']:not([class*='Divider'])"
        )
        if sidebar:
            sidebar_html = sidebar.html
            if sidebar_html:
                # Parse a copy to avoid modifying the original tree
                sidebar_tree = HTMLParser(sidebar_html)
                # Clean up noisy elements
                for svg in sidebar_tree.css("svg"):
                    svg.decompose()
                for btn in sidebar_tree.css(
                    "button, .js-zeroclipboard, [data-copy-feedback]"
                ):
                    btn.decompose()
                # Remove loading placeholders and error messages
                for elem in sidebar_tree.css(".is-error, .is-loading, .Skeleton"):
                    elem.decompose()
                cleaned_html = sidebar_tree.body.html if sidebar_tree.body else ""
                if cleaned_html:
                    fragments.append(cleaned_html)
        return self._unique_html_fragments(fragments)

    def _extract_footer_html(self, tree: HTMLParser) -> list[str]:
        """Extract footer HTML (typically minimal for GitHub repos)."""
        fragments: list[str] = []
        # GitHub repo footer is usually the global footer, which we don't want
        # Only extract if there's repo-specific footer content
        # For now, return empty to avoid noise
        return self._unique_html_fragments(fragments)

    def _unique_html_fragments(self, fragments: list[str]) -> list[str]:
        """Deduplicate HTML fragments."""
        seen: set[str] = set()
        result: list[str] = []
        for frag in fragments:
            key = frag[:100] if len(frag) >= 100 else frag
            if key not in seen:
                seen.add(key)
                result.append(frag)
        return result

    def _clean_readme_node(self, node: Any) -> None:
        """Clean README node in-place by removing noisy elements."""
        # Use css method which works on both HTMLParser and Tag nodes
        for svg in node.css("svg"):
            svg.decompose()
        for anchor in node.css("a.anchor"):
            anchor.decompose()
        for btn in node.css("button, .js-zeroclipboard, [data-copy-feedback]"):
            btn.decompose()

    def _extract_primary(
        self,
        *,
        raw_html: str,
        snapshot: GitHubSnapshot,
        url: str,
        preserve_html_tags: bool,
    ) -> PrimaryResult:
        # Extract body content (README for GitHub) - combine all fragments
        body_fragments = snapshot.semantic_html.get("body", [])
        markdown_parts: list[str] = []
        for fragment in body_fragments:
            if fragment:
                converted = self._html_to_markdown(
                    fragment, keep_html=preserve_html_tags
                )
                if converted:
                    markdown_parts.append(converted)

        body_markdown = "\n\n".join(markdown_parts)

        # Use trafilatura for metadata extraction
        trafilatura_result = self._extract_trafilatura(raw_html=raw_html, url=url)

        # Prefer body content if available, otherwise use trafilatura
        if body_markdown:
            return PrimaryResult(
                title=trafilatura_result.title if trafilatura_result else "",
                image=trafilatura_result.image if trafilatura_result else "",
                markdown=body_markdown,
                engine="readme",
            )

        if trafilatura_result:
            return trafilatura_result

        # Fallback to DOM extraction
        return self._extract_dom_fallback(
            snapshot=snapshot,
            preserve_html_tags=preserve_html_tags,
        )

    def _extract_trafilatura(self, *, raw_html: str, url: str) -> PrimaryResult | None:
        try:
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
        except Exception:  # noqa: BLE001
            return None

        payload: dict[str, Any] | None
        if isinstance(raw, dict):
            payload = raw
        elif raw is not None and callable(getattr(raw, "as_dict", None)):
            payload = raw.as_dict()
        else:
            payload = None

        if payload is None:
            return None

        markdown = str(payload.get("text") or payload.get("raw_text") or "").strip()
        if not markdown:
            return None

        image = clean_whitespace(str(payload.get("image") or ""))

        return PrimaryResult(
            title=clean_whitespace(str(payload.get("title") or "")),
            image=urljoin(url, image) if image else "",
            markdown=markdown,
            engine="trafilatura",
        )

    def _extract_dom_fallback(
        self,
        *,
        snapshot: GitHubSnapshot,
        preserve_html_tags: bool,
    ) -> PrimaryResult:
        # Try body extraction
        body_html = ""
        if snapshot.tree.body:
            body_html = str(snapshot.tree.body.html or "")

        markdown = self._html_to_markdown(body_html, keep_html=preserve_html_tags)

        return PrimaryResult(
            title=snapshot.repo_meta.full_name or "",
            image=self._get_meta_content(snapshot.tree, "og:image", use_property=True),
            markdown=markdown,
            engine="dom",
        )

    def _html_to_markdown(self, html: str, *, keep_html: bool) -> str:
        if not html:
            return ""

        if keep_html:
            html = re.sub(
                r"<svg[^>]*>.*?</svg>", "", html, flags=re.DOTALL | re.IGNORECASE
            )
            html = re.sub(
                r"<a\s+class=\"anchor\"[^>]*>.*?</a>",
                "",
                html,
                flags=re.DOTALL | re.IGNORECASE,
            )
            return html.strip()

        try:
            result = convert_with_tables(
                html,
                options=self._CONVERSION_OPTIONS,
                preprocessing=self._PREPROCESSING,
                metadata_config=self._METADATA_CONFIG,
            )
            markdown = str(result.get("content", "")).strip()
            markdown = re.sub(r"!\[SVG Image\]\([^)]*\)", "", markdown)
            markdown = re.sub(r"!SVG Image", "", markdown)
            markdown = re.sub(r"\n{3,}", "\n\n", markdown)
            return markdown.strip()
        except Exception:  # noqa: BLE001
            return ""

    def _render_document(
        self,
        *,
        snapshot: GitHubSnapshot,
        primary_result: PrimaryResult,
        selected_tags: set[ExtractContentTag],
        url: str,
        preserve_html_tags: bool,
    ) -> RenderResult:
        blocks: list[str] = []

        for tag in self._SEMANTIC_ORDER:
            if tag not in selected_tags:
                continue

            if tag == "body":
                # Body uses primary extraction result (README content)
                content = primary_result.markdown
            else:
                # Other sections are rendered from semantic_html
                fragments = snapshot.semantic_html.get(tag, [])
                rendered = [
                    self._html_to_markdown(frag, keep_html=preserve_html_tags)
                    for frag in fragments
                ]
                content = "\n\n".join(r for r in rendered if r).strip()

            if not content:
                continue

            blocks.append(self._wrap_block(tag, content, preserve_html_tags))

        return RenderResult(markdown="\n\n".join(blocks).strip())

    def _wrap_block(
        self, tag: ExtractContentTag, content: str, preserve_html_tags: bool
    ) -> str:
        if preserve_html_tags:
            return f'<section data-raysearch-tag="{tag}">\n{content}\n</section>'
        if tag == "body":
            return content
        if tag == "sidebar":
            return f"## Secondary Content\n\n{content}"
        return f"## {tag.capitalize()}\n\n{content}"

    def _build_stats(
        self,
        *,
        repo_meta: GitHubRepoMeta,
        markdown: str,
        detail: str,
        selected_tags: set[ExtractContentTag],
        engine: str,
    ) -> dict[str, int | float | str | bool]:
        stats: dict[str, int | float | str | bool] = {
            "primary_chars": len(markdown),
            "text_chars": len(markdown.replace("```", "").replace("`", "")),
            "detail": detail,
            "selected_sections": ",".join(sorted(selected_tags)),
            "engine_chain": engine,
            "github_owner": repo_meta.owner,
            "github_repo": repo_meta.repo,
            "github_stars": repo_meta.stars,
            "github_forks": repo_meta.forks,
            "github_language": repo_meta.language,
        }
        if repo_meta.is_archived:
            stats["github_archived"] = True
        return stats

    def _extract_links(self, *, snapshot: GitHubSnapshot, url: str) -> list[ExtractRef]:
        links: list[ExtractRef] = []
        seen: set[str] = set()

        # Extract from README
        readme = snapshot.tree.css_first("article.markdown-body")
        if readme:
            for a in readme.css("a[href]"):
                href = str(a.attributes.get("href", ""))
                if href and not href.startswith(("#", "javascript:")):
                    if href.startswith("/"):
                        href = urljoin("https://github.com", href)
                    elif not href.startswith(("http://", "https://")):
                        href = urljoin(url, href)
                    if href not in seen:
                        seen.add(href)
                        links.append(
                            ExtractRef(
                                url=href,
                                text=clean_whitespace(a.text(strip=True)),
                            )
                        )

        # Add repo links
        if snapshot.repo_meta.owner and snapshot.repo_meta.repo:
            repo_base = f"https://github.com/{snapshot.repo_meta.owner}/{snapshot.repo_meta.repo}"
            for path in ["/releases", "/issues", "/pulls", "/wiki", "/actions"]:
                link_url = repo_base + path
                if link_url not in seen:
                    seen.add(link_url)
                    links.append(ExtractRef(url=link_url, text=path.strip("/")))

        return links[:20]

    def _extract_images(
        self, *, snapshot: GitHubSnapshot, url: str
    ) -> list[ExtractRef]:
        images: list[ExtractRef] = []
        seen: set[str] = set()

        # Get README images
        readme = snapshot.tree.css_first("article.markdown-body")
        if readme:
            for img in readme.css("img[src]"):
                src = str(img.attributes.get("src", ""))
                if src:
                    if src.startswith("/"):
                        src = urljoin("https://github.com", src)
                    elif not src.startswith(("http://", "https://")):
                        src = urljoin(url, src)
                    if src not in seen:
                        seen.add(src)
                        alt = str(img.attributes.get("alt", ""))
                        images.append(ExtractRef(url=src, text=alt))

        # Add OG image
        og_image = self._get_meta_content(snapshot.tree, "og:image", use_property=True)
        if og_image and og_image not in seen:
            images.insert(0, ExtractRef(url=og_image, text="og:image"))

        return images[:10]


@dataclass(slots=True)
class PrimaryResult:
    """Result from primary extraction."""

    title: str = ""
    image: str = ""
    markdown: str = ""
    engine: str = "dom"


@dataclass(slots=True)
class RenderResult:
    """Result from rendering document sections."""

    markdown: str


__all__ = ["GitHubExtractor", "GitHubExtractorConfig", "GitHubRepoMeta"]
