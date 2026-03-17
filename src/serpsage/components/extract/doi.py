"""DOI specialized extractor for parsing Crossref API JSON data.

Extracts structured content from DOI metadata including:
- Title and authors
- Abstract (if available)
- Journal/venue information
- Publication date
- Citation count
- References and links

This extractor handles JSON content from the Crossref API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from typing_extensions import override

from serpsage.components.extract.base import (
    ExtractConfigBase,
    SpecializedExtractorBase,
)
from serpsage.models.components.extract import (
    ExtractContent,
    ExtractedDocument,
    ExtractMeta,
    ExtractRef,
    ExtractRefs,
    ExtractSpec,
    ExtractTrace,
)
from serpsage.utils import clean_whitespace


class DOIExtractorConfig(ExtractConfigBase):
    __setting_family__ = "extract"
    __setting_name__ = "doi"


@dataclass(slots=True)
class DOIMeta:
    """Extracted DOI metadata."""

    doi: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    journal: str = ""
    publisher: str = ""
    published_date: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    citation_count: int = 0
    references_count: int = 0
    article_type: str = ""
    issn: str = ""
    isbn: str = ""
    url: str = ""


class DOIExtractor(SpecializedExtractorBase[DOIExtractorConfig]):
    """Specialized extractor for DOI metadata from Crossref API.

    Extracts bibliographic information from Crossref JSON responses.
    """

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
        # First check content_kind is json
        if content_kind != "json":
            return False
        # Then check crawl_backend is doi
        return crawl_backend == "doi"

    @override
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None = None,
        crawl_backend: str = "curl_cffi",
        content_kind: Literal[
            "html", "pdf", "text", "markdown", "json", "binary", "unknown"
        ] = "unknown",
        content_options: ExtractSpec | None = None,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        """Extract content from Crossref API JSON response."""
        options = content_options or ExtractSpec()

        return await self._extract_from_json(
            url=url,
            content=content,
            options=options,
            collect_links=collect_links,
        )

    async def _extract_from_json(
        self,
        *,
        url: str,
        content: bytes,
        options: ExtractSpec,
        collect_links: bool,
    ) -> ExtractedDocument:
        """Extract content from Crossref API JSON data."""
        data = json.loads(content.decode("utf-8", errors="replace"))

        # Crossref wraps response in "message" key
        message = data.get("message", data)
        if not isinstance(message, dict):
            raise TypeError("Invalid Crossref JSON format")

        # Extract metadata
        doi_meta = self._extract_meta(message)

        # Build markdown
        markdown = self._build_markdown(
            message=message,
            doi_meta=doi_meta,
            options=options,
        )

        stats = self._build_stats(
            doi_meta=doi_meta,
            markdown=markdown,
            detail=options.detail,
        )

        links: list[ExtractRef] = []
        if collect_links:
            links = self._extract_links(message)

        return self._finalize_content(
            doc=ExtractedDocument(
                content=ExtractContent(markdown=markdown),
                meta=ExtractMeta(
                    title=doi_meta.title or doi_meta.doi or url,
                    author=", ".join(doi_meta.authors[:5]) if doi_meta.authors else "",
                    published_date=doi_meta.published_date,
                    favicon="https://www.crossref.org/favicon.ico",
                ),
                refs=ExtractRefs(links=links, images=[]),
                trace=ExtractTrace(
                    kind="json",
                    engine="doi:crossref_api",
                    stats=stats,
                ),
            ),
            content_options=options,
        )

    def _extract_meta(self, message: dict[str, Any]) -> DOIMeta:
        """Extract metadata from Crossref message."""
        meta = DOIMeta()

        # DOI
        meta.doi = clean_whitespace(str(message.get("DOI", "") or "")).lower()

        # Title (can be a list or string)
        title_raw = message.get("title")
        if isinstance(title_raw, list) and title_raw:
            meta.title = clean_whitespace(str(title_raw[0] or ""))
        elif isinstance(title_raw, str):
            meta.title = clean_whitespace(title_raw)

        # Authors
        meta.authors = self._extract_authors(message.get("author", []))

        # Abstract
        meta.abstract = self._extract_abstract(message.get("abstract"))

        # Journal/Container title
        container = message.get("container-title")
        if isinstance(container, list) and container:
            meta.journal = clean_whitespace(str(container[0] or ""))
        elif isinstance(container, str):
            meta.journal = clean_whitespace(container)

        # Publisher
        meta.publisher = clean_whitespace(str(message.get("publisher", "") or ""))

        # Publication date
        meta.published_date = self._extract_published_date(message)

        # Volume, issue, pages
        meta.volume = clean_whitespace(str(message.get("volume", "") or ""))
        meta.issue = clean_whitespace(str(message.get("issue", "") or ""))
        meta.pages = clean_whitespace(str(message.get("page", "") or ""))

        # Citation and reference counts
        meta.citation_count = int(message.get("is-referenced-by-count", 0) or 0)
        meta.references_count = int(message.get("references-count", 0) or 0)

        # Type
        meta.article_type = clean_whitespace(str(message.get("type", "") or ""))

        # ISSN/ISBN
        issn_list = message.get("ISSN", [])
        if isinstance(issn_list, list) and issn_list:
            meta.issn = clean_whitespace(str(issn_list[0] or ""))
        isbn_list = message.get("ISBN", [])
        if isinstance(isbn_list, list) and isbn_list:
            meta.isbn = clean_whitespace(str(isbn_list[0] or ""))

        # URL
        meta.url = clean_whitespace(str(message.get("URL", "") or ""))

        return meta

    def _extract_authors(self, authors_raw: Any) -> list[str]:
        """Extract author names from Crossref author list."""
        authors: list[str] = []

        if not isinstance(authors_raw, list):
            return authors

        for author in authors_raw:
            if not isinstance(author, dict):
                continue

            # Try given + family name
            given = clean_whitespace(str(author.get("given", "") or ""))
            family = clean_whitespace(str(author.get("family", "") or ""))

            if family:
                if given:
                    authors.append(f"{family}, {given}")
                else:
                    authors.append(family)
            elif given:
                authors.append(given)

        return authors

    def _extract_abstract(self, abstract_raw: Any) -> str:
        """Extract and clean abstract from Crossref data."""
        if not abstract_raw:
            return ""

        abstract = clean_whitespace(str(abstract_raw))

        # Remove JATS XML tags if present
        abstract = re.sub(r"<jats:p[^>]*>", "", abstract)
        abstract = re.sub(r"</jats:p>", "\n\n", abstract)
        abstract = re.sub(r"<jats:[^>]*>", "", abstract)
        abstract = re.sub(r"</jats:[^>]*>", "", abstract)
        abstract = re.sub(r"<[^>]+>", "", abstract)  # Remove any other HTML

        return clean_whitespace(abstract)

    def _extract_published_date(self, message: dict[str, Any]) -> str:
        """Extract publication date from Crossref message."""
        # Try published-print first
        for key in ("published-print", "published-online", "published", "created"):
            date_info = message.get(key)
            if isinstance(date_info, dict):
                date_parts = date_info.get("date-parts")
                if isinstance(date_parts, list) and date_parts:
                    parts = date_parts[0]
                    if isinstance(parts, list) and len(parts) >= 1:
                        year = int(parts[0])
                        month = int(parts[1]) if len(parts) >= 2 else 1
                        day = int(parts[2]) if len(parts) >= 3 else 1
                        try:
                            dt = datetime(year, month, day, tzinfo=UTC)
                            return dt.strftime("%Y-%m-%d")
                        except ValueError:
                            return str(year)

        # Fallback: try to find any date-like field
        for key in ("deposited", "indexed"):
            date_str = message.get(key, {}).get("date-time", "")
            if date_str:
                try:
                    dt = datetime.fromisoformat(date_str)
                    return dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    continue

        return ""

    def _build_markdown(
        self,
        *,
        message: dict[str, Any],
        doi_meta: DOIMeta,
        options: ExtractSpec,
    ) -> str:
        """Build markdown from Crossref data."""
        parts: list[str] = []

        # Metadata section
        meta_lines = self._build_metadata_lines(doi_meta)
        if meta_lines:
            parts.append("## Metadata")
            parts.append("")
            parts.extend(meta_lines)
            parts.append("")

        # Title
        if doi_meta.title:
            parts.append(f"# {doi_meta.title}")
            parts.append("")

        # Abstract
        if doi_meta.abstract and options.detail in ("standard", "full"):
            parts.append("## Abstract")
            parts.append("")
            parts.append(doi_meta.abstract)
            parts.append("")

        # References (if available and requested)
        if options.detail == "full":
            references = self._extract_references_list(message.get("reference", []))
            if references:
                parts.append("---")
                parts.append("")
                parts.append("## References")
                parts.append("")
                parts.extend(references[:50])  # Limit references
                if len(references) > 50:
                    parts.append("")
                    parts.append(f"... and {len(references) - 50} more references")

        return "\n".join(parts)

    def _build_metadata_lines(self, meta: DOIMeta) -> list[str]:
        """Build metadata lines for markdown."""
        lines: list[str] = []

        if meta.doi:
            lines.append(f"- **DOI**: [{meta.doi}](https://doi.org/{meta.doi})")

        if meta.authors:
            author_str = ", ".join(meta.authors[:10])
            if len(meta.authors) > 10:
                author_str += f" et al. ({len(meta.authors)} authors)"
            lines.append(f"- **Authors**: {author_str}")

        if meta.journal:
            lines.append(f"- **Journal**: {meta.journal}")

        if meta.publisher and not meta.journal:
            lines.append(f"- **Publisher**: {meta.publisher}")

        if meta.published_date:
            lines.append(f"- **Published**: {meta.published_date}")

        # Volume/issue/pages
        venue_parts: list[str] = []
        if meta.volume:
            venue_parts.append(f"Vol. {meta.volume}")
        if meta.issue:
            venue_parts.append(f"No. {meta.issue}")
        if meta.pages:
            venue_parts.append(f"pp. {meta.pages}")
        if venue_parts:
            lines.append(f"- **Volume/Issue**: {', '.join(venue_parts)}")

        if meta.article_type:
            lines.append(f"- **Type**: {self._format_article_type(meta.article_type)}")

        if meta.citation_count > 0:
            lines.append(f"- **Citations**: {meta.citation_count}")

        if meta.references_count > 0:
            lines.append(f"- **References**: {meta.references_count}")

        if meta.issn:
            lines.append(f"- **ISSN**: {meta.issn}")

        return lines

    def _format_article_type(self, article_type: str) -> str:
        """Format article type for display."""
        type_map = {
            "journal-article": "Journal Article",
            "book": "Book",
            "book-chapter": "Book Chapter",
            "proceedings-article": "Conference Paper",
            "conference-paper": "Conference Paper",
            "dissertation": "Dissertation",
            "report": "Report",
            "preprint": "Preprint",
            "posted-content": "Online Content",
            "dataset": "Dataset",
            "peer-review": "Peer Review",
            "standard": "Standard",
        }
        return type_map.get(article_type.lower(), article_type.title())

    def _extract_references_list(self, references_raw: Any) -> list[str]:
        """Extract formatted references from Crossref reference list."""
        refs: list[str] = []

        if not isinstance(references_raw, list):
            return refs

        for ref in references_raw:
            if not isinstance(ref, dict):
                continue

            parts: list[str] = []

            # Author
            author = clean_whitespace(str(ref.get("author", "") or ""))
            if author:
                parts.append(author)

            # Year
            year = clean_whitespace(str(ref.get("year", "") or ""))
            if year:
                parts.append(f"({year})")

            # Title
            title = clean_whitespace(
                str(ref.get("article-title", "") or ref.get("title", "") or "")
            )
            if title:
                parts.append(f"**{title}**")

            # Journal
            journal = clean_whitespace(str(ref.get("journal-title", "") or ""))
            if journal:
                parts.append(f"*{journal}*")

            # DOI
            doi = clean_whitespace(str(ref.get("DOI", "") or ""))
            if doi:
                parts.append(f"[doi:{doi}](https://doi.org/{doi})")

            if parts:
                refs.append("- " + " ".join(parts))

        return refs

    def _extract_links(self, message: dict[str, Any]) -> list[ExtractRef]:
        """Extract links from Crossref data with open access priority."""
        links: list[ExtractRef] = []
        seen: set[str] = set()

        # Collect full text links
        full_text_links: list[ExtractRef] = []
        for link_info in message.get("link", []):
            if isinstance(link_info, dict):
                full_text_url = clean_whitespace(str(link_info.get("URL", "") or ""))
                content_type = clean_whitespace(
                    str(link_info.get("content-type", "") or "")
                )
                if full_text_url and full_text_url not in seen:
                    # Convert PDF URLs to HTML URLs for better crawling success
                    if full_text_url.endswith("/pdf"):
                        html_url = full_text_url[:-4]
                        if html_url not in seen:
                            seen.add(html_url)
                            full_text_links.append(
                                ExtractRef(url=html_url, text="Full Text (HTML)")
                            )
                    else:
                        seen.add(full_text_url)
                        if (
                            "pdf" in full_text_url.lower()
                            or "pdf" in content_type.lower()
                        ):
                            full_text_links.append(
                                ExtractRef(url=full_text_url, text="PDF Full Text")
                            )
                        else:
                            full_text_links.append(
                                ExtractRef(url=full_text_url, text="Full Text")
                            )

        # Sort full text links by accessibility priority
        full_text_links = self._sort_links_by_priority(full_text_links)
        links.extend(full_text_links)

        # DOI link (usually resolves to publisher page)
        doi = clean_whitespace(str(message.get("DOI", "") or ""))
        if doi:
            doi_url = f"https://doi.org/{doi}"
            if doi_url not in seen:
                seen.add(doi_url)
                links.append(ExtractRef(url=doi_url, text="DOI"))

        # License links
        for license_info in message.get("license", []):
            if isinstance(license_info, dict):
                license_url = clean_whitespace(str(license_info.get("URL", "") or ""))
                if license_url and license_url not in seen:
                    seen.add(license_url)
                    links.append(ExtractRef(url=license_url, text="License"))

        # Author ORCID
        for author in message.get("author", []):
            if isinstance(author, dict):
                orcid = clean_whitespace(str(author.get("ORCID", "") or ""))
                if orcid and orcid not in seen:
                    seen.add(orcid)
                    links.append(ExtractRef(url=orcid, text="ORCID"))

        # Funder links
        for funder in message.get("funder", []):
            if isinstance(funder, dict):
                funder_doi = clean_whitespace(str(funder.get("DOI", "") or ""))
                if funder_doi and funder_doi not in seen:
                    seen.add(funder_doi)
                    links.append(
                        ExtractRef(
                            url=funder_doi, text=f"Funder: {funder.get('name', '')}"
                        )
                    )

        return links[: self.config.link_max_count]

    def _sort_links_by_priority(self, links: list[ExtractRef]) -> list[ExtractRef]:
        """Sort links by accessibility priority.

        Open access and easily crawlable platforms are prioritized.
        """
        # High priority: Open access platforms with good API/HTML access
        high_priority_domains = (
            "plos.org",  # PLoS - open access, good HTML
            "ncbi.nlm.nih.gov",  # PubMed Central - open access
            "arxiv.org",  # arXiv - open access
            "biorxiv.org",  # bioRxiv - open access
            "medrxiv.org",  # medRxiv - open access
            "frontiersin.org",  # Frontiers - open access
            "sciencedirect.com",  # ScienceDirect (some open)
            "nature.com",  # Nature (some open)
        )

        # Low priority: Publishers known to block automated access
        low_priority_domains = (
            "mdpi.com",  # MDPI - returns 403 for bots
            "elsevier.com",  # Elsevier - paywall/bot protection
            "springer.com",  # Springer - paywall/bot protection
            "wiley.com",  # Wiley - paywall/bot protection
            "tandfonline.com",  # Taylor & Francis - paywall
            "ieee.org",  # IEEE - paywall
            "acm.org",  # ACM - paywall
        )

        high: list[ExtractRef] = []
        medium: list[ExtractRef] = []
        low: list[ExtractRef] = []

        for link in links:
            url_lower = link.url.lower()
            if any(domain in url_lower for domain in high_priority_domains):
                high.append(link)
            elif any(domain in url_lower for domain in low_priority_domains):
                low.append(link)
            else:
                medium.append(link)

        return high + medium + low

    def _build_stats(
        self,
        *,
        doi_meta: DOIMeta,
        markdown: str,
        detail: str,
    ) -> dict[str, int | float | str | bool]:
        """Build stats dictionary."""
        stats: dict[str, int | float | str | bool] = {
            "primary_chars": len(markdown),
            "text_chars": len(markdown.replace("```", "").replace("`", "")),
            "detail": detail,
            "engine": "doi_crossref",
        }
        if doi_meta.doi:
            stats["doi"] = doi_meta.doi
        if doi_meta.citation_count:
            stats["citation_count"] = doi_meta.citation_count
        if doi_meta.references_count:
            stats["references_count"] = doi_meta.references_count
        if doi_meta.article_type:
            stats["article_type"] = doi_meta.article_type
        return stats


__all__ = ["DOIExtractor", "DOIExtractorConfig", "DOIMeta"]
