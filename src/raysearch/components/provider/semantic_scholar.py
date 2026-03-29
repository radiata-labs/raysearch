# SPDX-License-Identifier: AGPL-3.0-or-later
"""Semantic Scholar provides AI-assisted academic search and paper discovery.

This provider keeps the same high-value request flow as the SearXNG engine:

- fetch the current ``X-S2-UI-Version`` from the main site
- cache that UI version briefly in-process
- send the search request as a JSON POST to ``/api/1/search``
- parse paper metadata from the returned JSON payload

Configuration
=============

Example configuration in this project:

.. code:: yaml

   semantic_scholar:
     enabled: true
     base_url: https://www.semanticscholar.org/api/1/search
     site_base_url: https://www.semanticscholar.org
     user_agent: raysearch-semantic-scholar-provider/1.0
     results_per_page: 10

Notes
=====

- Semantic Scholar's search API expects a current UI version header, so this
  provider resolves and caches ``X-S2-UI-Version`` before sending search calls.
- The provider prefers the primary paper link, then alternate links, and finally
  falls back to the canonical Semantic Scholar paper page.
- Citation and field-of-study metadata are folded into the snippet to preserve
  useful ranking signals for later stages.

Implementation
==============

The provider keeps the UI-version handshake separate from result fetching so the
main search path stays small and deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from typing_extensions import override

import httpx
from bs4 import BeautifulSoup
from pydantic import field_validator

from raysearch.components.http.base import HttpClientBase
from raysearch.components.provider.base import (
    ProviderConfigBase,
    ProviderMeta,
    SearchProviderBase,
)
from raysearch.dependencies import Depends
from raysearch.models.components.provider import SearchProviderResult
from raysearch.utils import clean_whitespace, normalize_iso8601_string, strip_html

_DEFAULT_SEMANTIC_SCHOLAR_BASE_URL = "https://www.semanticscholar.org"
_DEFAULT_SEMANTIC_SCHOLAR_SEARCH_URL = "https://www.semanticscholar.org/api/1/search"
_DEFAULT_SEMANTIC_SCHOLAR_USER_AGENT = "raysearch-semantic-scholar-provider/1.0"
_DEFAULT_SEMANTIC_SCHOLAR_RESULTS_PER_PAGE = 10
_MAX_SEMANTIC_SCHOLAR_RESULTS_PER_PAGE = 50
_MAX_SEMANTIC_SCHOLAR_SCAN_PAGES = 5


class SemanticScholarProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "semantic_scholar"

    base_url: str = _DEFAULT_SEMANTIC_SCHOLAR_SEARCH_URL
    site_base_url: str = _DEFAULT_SEMANTIC_SCHOLAR_BASE_URL
    user_agent: str = _DEFAULT_SEMANTIC_SCHOLAR_USER_AGENT
    results_per_page: int = _DEFAULT_SEMANTIC_SCHOLAR_RESULTS_PER_PAGE

    @field_validator("site_base_url", "user_agent")
    @classmethod
    def _normalize_text_fields(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("semantic_scholar results_per_page must be > 0")
        if size > _MAX_SEMANTIC_SCHOLAR_RESULTS_PER_PAGE:
            raise ValueError(
                "semantic_scholar results_per_page must be <= "
                f"{_MAX_SEMANTIC_SCHOLAR_RESULTS_PER_PAGE}"
            )
        return size

    @classmethod
    @override
    def inject_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> dict[str, Any]:
        payload = dict(raw)
        if env.get("SEMANTIC_SCHOLAR_BASE_URL"):
            payload["base_url"] = env["SEMANTIC_SCHOLAR_BASE_URL"]
        if env.get("SEMANTIC_SCHOLAR_SITE_BASE_URL"):
            payload["site_base_url"] = env["SEMANTIC_SCHOLAR_SITE_BASE_URL"]
        return payload


class SemanticScholarProvider(
    SearchProviderBase[SemanticScholarProviderConfig],
    meta=ProviderMeta(
        name="semantic_scholar",
        website="https://www.semanticscholar.org/",
        description="Academic paper search through Semantic Scholar with metadata-rich results across papers, authors, and venues.",
        preference="Prefer literature discovery, paper lookup, citation-oriented searches, and academic topic exploration beyond preprint-only sources.",
        categories=["science", "scientific publications"],
    ),
):
    http: HttpClientBase = Depends()

    _ui_version_cache: str = ""
    _ui_version_expires_at: datetime = datetime.fromtimestamp(0, tz=UTC)

    @override
    async def _asearch(
        self,
        *,
        query: str,
        limit: int | None = None,
        language: str = "",
        location: str = "",
        moderation: bool = True,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        per_page = self._coerce_per_page(
            limit if limit is not None else self.config.results_per_page
        )
        should_backfill = bool(start_published_date or end_published_date)
        max_pages = _MAX_SEMANTIC_SCHOLAR_SCAN_PAGES if should_backfill else 1
        page = self._coerce_page(kwargs.get("page"))
        request_size = (
            self._initial_dated_fetch_limit(
                target_limit=per_page,
                max_limit=_MAX_SEMANTIC_SCHOLAR_RESULTS_PER_PAGE,
            )
            if should_backfill
            else per_page
        )
        results: list[SearchProviderResult] = []

        for page_index in range(max_pages):
            resp = await self.http.client.post(
                str(self.config.base_url),
                json=self._build_payload(
                    query=normalized_query,
                    page=page + page_index,
                    page_size=request_size if page_index == 0 else per_page,
                ),
                headers=await self._build_headers(),
                timeout=httpx.Timeout(self.config.timeout_s),
                follow_redirects=bool(self.config.allow_redirects),
            )
            resp.raise_for_status()
            batch = self._parse_results(resp.json().get("results"))
            if not batch:
                break
            batch = self._filter_results_by_published_date(
                results=batch,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
            )
            results.extend(batch)
            if len(results) >= per_page:
                break
        return results[:per_page]

    async def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        headers["User-Agent"] = (
            clean_whitespace(str(self.config.user_agent or ""))
            or _DEFAULT_SEMANTIC_SCHOLAR_USER_AGENT
        )
        headers["X-S2-UI-Version"] = await self._get_ui_version()
        headers["X-S2-Client"] = "webapp-browser"
        return headers

    def _build_payload(
        self, *, query: str, page: int, page_size: int
    ) -> dict[str, Any]:
        return {
            "queryString": query,
            "page": max(1, int(page)),
            "pageSize": max(1, int(page_size)),
            "sort": "relevance",
            "getQuerySuggestions": False,
            "authors": [],
            "coAuthors": [],
            "venues": [],
            "performTitleMatch": True,
        }

    async def _get_ui_version(self) -> str:
        now_utc = datetime.now(tz=UTC)
        if (
            self.__class__._ui_version_cache
            and now_utc < self.__class__._ui_version_expires_at
        ):
            return self.__class__._ui_version_cache
        resp = await self.http.client.get(
            str(self.config.site_base_url),
            headers={
                "User-Agent": (
                    clean_whitespace(str(self.config.user_agent or ""))
                    or _DEFAULT_SEMANTIC_SCHOLAR_USER_AGENT
                )
            },
            timeout=httpx.Timeout(self.config.timeout_s),
            follow_redirects=bool(self.config.allow_redirects),
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        node = soup.select_one('meta[name="s2-ui-version"]')
        ui_version = clean_whitespace(str(node.get("content", ""))) if node else ""
        if not ui_version:
            raise RuntimeError("failed to determine Semantic Scholar UI version")
        self.__class__._ui_version_cache = ui_version
        self.__class__._ui_version_expires_at = now_utc + timedelta(minutes=5)
        return ui_version

    def _parse_results(self, raw_results: Any) -> list[SearchProviderResult]:
        results: list[SearchProviderResult] = []
        for item in raw_results if isinstance(raw_results, list) else []:
            if not isinstance(item, dict):
                continue
            url = self._extract_url(item)
            title = clean_whitespace(str(item.get("title", {}).get("text") or ""))
            if not url or not title:
                continue
            abstract = strip_html(
                clean_whitespace(str(item.get("paperAbstract", {}).get("text") or ""))
            )
            authors = self._extract_authors(item)
            author_str = ", ".join(authors[:5]) if authors else ""
            results.append(
                SearchProviderResult(
                    url=url,
                    title=title,
                    snippet=self._build_snippet(item),
                    engine=self.config.name,
                    published_date=self._parse_published_date(item.get("pubDate")),
                    pre_fetched_content=abstract,
                    pre_fetched_author=author_str,
                )
            )
        return results

    def _extract_url(self, item: dict[str, Any]) -> str:
        primary_link = item.get("primaryPaperLink")
        if isinstance(primary_link, dict):
            url = clean_whitespace(str(primary_link.get("url") or ""))
            if url:
                return url
        links = item.get("links")
        if isinstance(links, list):
            for raw_link in links:
                url = clean_whitespace(str(raw_link or ""))
                if url:
                    return url
        alternate_links = item.get("alternatePaperLinks")
        if isinstance(alternate_links, list):
            for raw_link in alternate_links:
                if not isinstance(raw_link, dict):
                    continue
                url = clean_whitespace(str(raw_link.get("url") or ""))
                if url:
                    return url
        paper_id = clean_whitespace(str(item.get("id") or ""))
        if paper_id:
            return f"{self.config.site_base_url.rstrip('/')}/paper/{paper_id}"
        return ""

    def _build_snippet(self, item: dict[str, Any]) -> str:
        parts: list[str] = []
        abstract = strip_html(
            clean_whitespace(str(item.get("paperAbstract", {}).get("text") or ""))
        )
        if abstract:
            parts.append(abstract)
        authors = self._extract_authors(item)
        if authors:
            parts.append(f"Authors: {', '.join(authors[:4])}")
        journal = clean_whitespace(
            str(
                item.get("venue", {}).get("text")
                or item.get("journal", {}).get("name")
                or ""
            )
        )
        if journal:
            parts.append(journal)
        fields = item.get("fieldsOfStudy")
        if isinstance(fields, list):
            tags = [clean_whitespace(str(field or "")) for field in fields]
            tags = [tag for tag in tags if tag]
            if tags:
                parts.append(f"Fields: {', '.join(tags[:4])}")
        comments = self._extract_comments(item)
        if comments:
            parts.append(comments)
        return clean_whitespace(" / ".join(parts))

    def _extract_authors(self, item: dict[str, Any]) -> list[str]:
        authors: list[str] = []
        for raw_author in item.get("authors", []):
            if isinstance(raw_author, list):
                for nested in raw_author:
                    if not isinstance(nested, dict):
                        continue
                    name = clean_whitespace(str(nested.get("name") or ""))
                    if name:
                        authors.append(name)
                        break
            elif isinstance(raw_author, dict):
                name = clean_whitespace(str(raw_author.get("name") or ""))
                if name:
                    authors.append(name)
        return authors

    def _extract_comments(self, item: dict[str, Any]) -> str:
        citation_stats = item.get("citationStats")
        if not isinstance(citation_stats, dict):
            return ""
        num_citations = citation_stats.get("numCitations")
        first_year = citation_stats.get("firstCitationVelocityYear")
        last_year = citation_stats.get("lastCitationVelocityYear")
        if isinstance(num_citations, int) and first_year and last_year:
            return f"{num_citations} citations from {first_year} to {last_year}"
        if isinstance(num_citations, int):
            return f"{num_citations} citations"
        return ""

    def _parse_published_date(self, value: Any) -> str:
        token = clean_whitespace(str(value or ""))
        if not token:
            return ""
        try:
            return normalize_iso8601_string(token)
        except ValueError:
            return ""

    def _coerce_per_page(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(_MAX_SEMANTIC_SCHOLAR_RESULTS_PER_PAGE, size))

    def _coerce_page(self, value: Any) -> int:
        try:
            page = int(value)
        except Exception:
            return 1
        return max(1, page)


__all__ = ["SemanticScholarProvider", "SemanticScholarProviderConfig"]
