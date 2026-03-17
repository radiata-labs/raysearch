from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from typing_extensions import override
from urllib.parse import urljoin, urlparse, urlunparse

from selectolax.parser import HTMLParser

from serpsage.components.extract.base import (
    ExtractConfigBase,
    SpecializedExtractorBase,
)
from serpsage.components.extract.html import HtmlExtractor
from serpsage.dependencies import Depends
from serpsage.models.components.extract import (
    ExtractedDocument,
    ExtractRef,
    ExtractSpec,
)


class PaperExtractorConfig(ExtractConfigBase):
    __setting_family__ = "extract"
    __setting_name__ = "paper"


class PaperExtractor(SpecializedExtractorBase[PaperExtractorConfig]):
    """Specialized extractor for academic paper pages.

    Detects paper pages by checking for academic meta tags following standard
    protocols (Google Scholar/Highwire Press, Dublin Core) and extracts
    structured metadata from these standardized tags.

    Supported metadata protocols:
    - Google Scholar/Highwire Press: citation_title, citation_author,
      citation_date, citation_abstract, citation_pdf_url, citation_doi, etc.
    - Dublin Core: dc.type, dc.identifier
    - Open Graph: og:title, og:description (as fallback)
    - HTML link elements: link[rel="alternate"] with document MIME types

    Extracted fields are mapped to ExtractedDocument:
    - meta.title: citation_title (or og:title fallback)
    - meta.author: citation_author values joined with "; "
    - meta.published_date: citation_date
    - content.abstract_text: citation_abstract
    """

    html_extractor: HtmlExtractor = Depends()

    # Meta tag names that indicate an academic paper (Google Scholar/Highwire)
    _PAPER_INDICATOR_TAGS: set[str] = {
        "citation_title",
        "citation_pdf_url",
        "citation_doi",
        "citation_arxiv_id",
        "dc.type",
    }

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

        if not content:
            return False

        # Check for paper indicator meta tags
        try:
            tree = HTMLParser(content)
        except Exception:  # noqa: BLE001
            return False

        for meta in tree.css("meta[name]"):
            name = str(meta.attributes.get("name", "")).lower()
            if name in cls._PAPER_INDICATOR_TAGS:
                return True

        return False

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
        doc = await self.html_extractor.extract(
            url=url,
            content=content,
            content_type=content_type,
            crawl_backend=crawl_backend,
            content_kind=content_kind,
            content_options=content_options,
            collect_links=collect_links,
            collect_images=collect_images,
        )

        # Enrich with paper-specific metadata from standardized meta tags
        paper_meta = self._extract_paper_meta(content=content)
        if paper_meta:
            doc = self._apply_paper_meta(doc=doc, paper_meta=paper_meta)

        if not collect_links:
            return doc

        document_urls = self._extract_document_urls(content=content, base_url=url)
        if not document_urls:
            return doc

        reordered = self._reorder_links(
            links=list(doc.refs.links or []),
            document_urls=document_urls,
        )

        return doc.model_copy(
            update={"refs": doc.refs.model_copy(update={"links": reordered})}
        )

    @dataclass
    class _PaperMeta:
        """Extracted paper metadata from standardized meta tags."""

        title: str = ""
        authors: list[str] = None  # type: ignore[assignment]
        published_date: str = ""
        abstract: str = ""

        def __post_init__(self) -> None:
            if self.authors is None:
                self.authors = []

    def _extract_paper_meta(self, *, content: bytes) -> _PaperMeta | None:
        """Extract paper metadata from standardized academic meta tags.

        Sources (in priority order):
        1. Google Scholar/Highwire Press: citation_* tags
        2. Dublin Core: dc.* tags
        3. Open Graph: og:* tags (as fallback)
        """
        try:
            tree = HTMLParser(content)
        except Exception:  # noqa: BLE001
            return None

        meta = self._PaperMeta()

        # Extract title: citation_title > og:title
        title = self._get_meta_content(tree, "citation_title")
        if not title:
            title = self._get_meta_content(tree, "og:title", use_property=True)
        meta.title = title

        # Extract authors: citation_author (multiple values)
        authors: list[str] = []
        for node in tree.css("meta[name='citation_author']"):
            author = str(node.attributes.get("content", "")).strip()
            if author:
                authors.append(author)
        meta.authors = authors

        # Extract date: citation_date > citation_publication_date
        date = self._get_meta_content(tree, "citation_date")
        if not date:
            date = self._get_meta_content(tree, "citation_publication_date")
        meta.published_date = date

        # Extract abstract: citation_abstract
        meta.abstract = self._get_meta_content(tree, "citation_abstract")

        # Return None if no paper-specific metadata found
        if not any([meta.title, meta.authors, meta.published_date, meta.abstract]):
            return None

        return meta

    def _get_meta_content(
        self,
        tree: HTMLParser,
        name: str,
        *,
        use_property: bool = False,
    ) -> str:
        """Get content from a meta tag by name or property."""
        if use_property:
            selector = f"meta[property='{name}']"
        else:
            selector = f"meta[name='{name}']"
        node = tree.css_first(selector)
        if node is None:
            return ""
        return str(node.attributes.get("content", "")).strip()

    def _apply_paper_meta(
        self,
        *,
        doc: ExtractedDocument,
        paper_meta: _PaperMeta,
    ) -> ExtractedDocument:
        """Apply extracted paper metadata to the document.

        Title, authors, and published_date are always overwritten from paper meta.
        Abstract is only set if original abstract_text is empty (non-destructive).
        """
        meta_updates: dict[str, Any] = {}
        content_updates: dict[str, Any] = {}

        # Title: always overwrite from paper meta
        if paper_meta.title:
            meta_updates["title"] = paper_meta.title

        # Authors: always overwrite, join with "; " if multiple
        if paper_meta.authors:
            meta_updates["author"] = "; ".join(paper_meta.authors)

        # Published date: always overwrite
        if paper_meta.published_date:
            meta_updates["published_date"] = paper_meta.published_date

        # Abstract: non-destructive, only set if empty
        if paper_meta.abstract and not doc.content.abstract_text:
            content_updates["abstract_text"] = paper_meta.abstract

        # Apply updates
        new_meta = (
            doc.meta.model_copy(update=meta_updates) if meta_updates else doc.meta
        )
        new_content = (
            doc.content.model_copy(update=content_updates)
            if content_updates
            else doc.content
        )

        return doc.model_copy(
            update={
                "meta": new_meta,
                "content": new_content,
            }
        )

    def _extract_document_urls(self, *, content: bytes, base_url: str) -> set[str]:
        """Extract document URLs from standardized academic metadata.

        Sources (in order of reliability):
        1. Google Scholar/Highwire: citation_pdf_url, citation_fulltext_html_url
        2. Dublin Core: dc.identifier with URL value
        3. HTML: link[rel="alternate"] with document MIME type
        """
        urls: set[str] = set()
        try:
            tree = HTMLParser(content)
        except Exception:  # noqa: BLE001
            return urls

        # Google Scholar / Highwire Press meta tags
        for meta in tree.css("meta[name]"):
            name = str(meta.attributes.get("name", "")).lower()
            if name.startswith("citation_") and name.endswith("_url"):
                href = str(meta.attributes.get("content", "")).strip()
                if href:
                    normalized = self._normalize_url(href, base_url)
                    if normalized:
                        urls.add(normalized)

        # Dublin Core: dc.identifier
        for meta in tree.css(
            'meta[name="dc.identifier"], meta[property="dc.identifier"]'
        ):
            identifier = str(meta.attributes.get("content", "")).strip()
            if identifier.lower().startswith(("http://", "https://", "/")):
                normalized = self._normalize_url(identifier, base_url)
                if normalized:
                    urls.add(normalized)

        # HTML link elements with document MIME types
        for link in tree.css('link[rel="alternate"]'):
            link_type = str(link.attributes.get("type", "")).lower()
            href = str(link.attributes.get("href", "")).strip()
            if link_type in ("application/pdf", "text/html", "application/xhtml+xml"):
                normalized = self._normalize_url(href, base_url)
                if normalized:
                    urls.add(normalized)

        return urls

    def _normalize_url(self, href: str, base_url: str) -> str | None:
        """Normalize a URL relative to a base URL."""
        raw = (href or "").strip()
        if not raw or raw.lower().startswith(
            ("javascript:", "mailto:", "tel:", "data:")
        ):
            return None
        try:
            parsed = urlparse(urljoin(base_url, raw))
        except Exception:  # noqa: BLE001
            return None
        if parsed.scheme not in {"http", "https"}:
            return None
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    def _reorder_links(
        self,
        *,
        links: list[ExtractRef],
        document_urls: set[str],
    ) -> list[ExtractRef]:
        """Reorder links with document URLs first.

        Uses URL path matching to handle variations like version suffixes.
        """
        if not document_urls:
            return links

        # Extract paths for fuzzy matching
        doc_paths = self._extract_paths(document_urls)

        doc_links: list[ExtractRef] = []
        other_links: list[ExtractRef] = []

        for link in links:
            if self._is_document_url(link.url, document_urls, doc_paths):
                doc_links.append(link)
            else:
                other_links.append(link)

        return doc_links + other_links

    def _extract_paths(self, urls: set[str]) -> set[str]:
        """Extract URL paths for matching."""
        paths: set[str] = set()
        for url in urls:
            try:
                parsed = urlparse(url)
                path = parsed.path.rstrip("/")
                if path:
                    paths.add(path)
            except Exception:  # noqa: BLE001, S112
                pass
        return paths

    def _is_document_url(
        self,
        url: str,
        exact_urls: set[str],
        doc_paths: set[str],
    ) -> bool:
        """Check if URL matches a document URL.

        Matches:
        1. Exact URL match
        2. Path prefix with version suffix (e.g., /pdf/ID matches /pdf/IDv1)
        3. Same ID in different format paths (e.g., /pdf/ID matches /html/ID)
        """
        if url in exact_urls:
            return True

        try:
            parsed = urlparse(url)
            url_path = parsed.path.rstrip("/")
        except Exception:  # noqa: BLE001
            return False

        for doc_path in doc_paths:
            # Prefix match with version suffix
            if url_path.startswith(doc_path):
                suffix = url_path[len(doc_path) :]
                if not suffix:
                    return True
                # Version suffix: v1, v2, etc.
                if suffix.startswith("v") and suffix[1:].isdigit():
                    return True

            # Same ID, different format directory
            # e.g., /pdf/2603.08092 and /html/2603.08092v1
            parts = doc_path.rsplit("/", 1)
            if len(parts) == 2:
                doc_dir, doc_id = parts
                url_parts = url_path.rsplit("/", 1)
                if len(url_parts) == 2:
                    url_dir, url_id = url_parts
                    # Different format directories, same ID base
                    if doc_dir != url_dir and doc_id:
                        # Strip version suffix for comparison
                        base_id = doc_id.split("v")[0] if "v" in doc_id else doc_id
                        url_base = url_id.split("v")[0] if "v" in url_id else url_id
                        if base_id and base_id == url_base:
                            return True

        return False


__all__ = ["PaperExtractor", "PaperExtractorConfig"]
