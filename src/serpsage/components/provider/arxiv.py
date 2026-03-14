"""arXiv search provider based on the public Atom API.

This provider follows the same practical shape as an arXiv engine in a search
stack:

- it queries the public Atom feed endpoint
- it supports configurable search prefixes such as ``all`` and ``ti``
- it parses entry metadata from the returned XML feed
- it can backfill across several pages when date filtering is requested

Configuration
=============

Example configuration in this project:

.. code:: yaml

   arxiv:
     enabled: true
     base_url: https://export.arxiv.org/api/query
     user_agent: serpsage-arxiv-provider/1.0
     search_prefix: all
     results_per_page: 10

Notes
=====

- arXiv returns Atom XML rather than JSON or HTML.
- The provider uses ``published`` as its normalized ``published_date`` source.
- When date bounds are present, the provider fetches a wider first page and then
  filters locally to improve recall.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import field_validator

from serpsage.components.http.base import HttpClientBase
from serpsage.components.provider.base import (
    ProviderConfigBase,
    ProviderMeta,
    SearchProviderBase,
)
from serpsage.dependencies import Depends
from serpsage.models.components.provider import SearchProviderResult
from serpsage.utils import clean_whitespace, normalize_iso8601_string

_DEFAULT_ARXIV_BASE_URL = "https://export.arxiv.org/api/query"
_DEFAULT_ARXIV_USER_AGENT = "serpsage-arxiv-provider/1.0"
_DEFAULT_ARXIV_RESULTS_PER_PAGE = 10
_MAX_ARXIV_RESULTS_PER_PAGE = 100
_MAX_ARXIV_DATE_SCAN_PAGES = 5


class ArxivProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "arxiv"

    base_url: str = _DEFAULT_ARXIV_BASE_URL
    user_agent: str = _DEFAULT_ARXIV_USER_AGENT
    search_prefix: str = "all"
    results_per_page: int = _DEFAULT_ARXIV_RESULTS_PER_PAGE

    @field_validator("user_agent", "search_prefix")
    @classmethod
    def _normalize_text_fields(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("arxiv results_per_page must be > 0")
        if size > _MAX_ARXIV_RESULTS_PER_PAGE:
            raise ValueError(
                f"arxiv results_per_page must be <= {_MAX_ARXIV_RESULTS_PER_PAGE}"
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
        if env.get("ARXIV_BASE_URL"):
            payload["base_url"] = env["ARXIV_BASE_URL"]
        return payload


class ArxivProvider(
    SearchProviderBase[ArxivProviderConfig],
    meta=ProviderMeta(
        name="arxiv",
        website="https://arxiv.org/",
        description="Academic paper search across arXiv preprints in science, mathematics, and computer science.",
        preference="Prefer research-style queries with technical terms, paper topics, methods, models, and scientific subject keywords.",
        categories=["science", "scientific publications"],
    ),
):
    http: HttpClientBase = Depends()

    @override
    async def _asearch(
        self,
        *,
        query: str,
        limit: int | None = None,
        locale: str = "",
        moderation: bool = True,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        cfg = self.config
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        per_page = self._coerce_per_page(
            limit if limit is not None else cfg.results_per_page
        )
        headers = self._build_headers()
        search_prefix = self._resolve_search_prefix(kwargs.get("search_prefix"))
        start = self._coerce_start(kwargs.get("start"))
        should_backfill = bool(start_published_date or end_published_date)
        max_pages = _MAX_ARXIV_DATE_SCAN_PAGES if should_backfill else 1
        initial_fetch_size = (
            self._initial_dated_fetch_limit(
                target_limit=per_page,
                max_limit=_MAX_ARXIV_RESULTS_PER_PAGE,
            )
            if should_backfill
            else per_page
        )
        results: list[SearchProviderResult] = []
        next_start = start

        for page_index in range(max_pages):
            fetch_size = initial_fetch_size if page_index == 0 else per_page
            resp = await self.http.client.get(
                str(cfg.base_url),
                params=self._build_params(
                    query=normalized_query,
                    search_prefix=search_prefix,
                    start=next_start,
                    per_page=fetch_size,
                ),
                headers=headers,
                timeout=httpx.Timeout(cfg.timeout_s),
                follow_redirects=bool(cfg.allow_redirects),
            )
            resp.raise_for_status()
            batch = self._parse_feed(resp.content)
            if not batch:
                break
            next_start += fetch_size
            results.extend(
                self._filter_results_by_published_date(
                    results=batch,
                    start_published_date=start_published_date,
                    end_published_date=end_published_date,
                )
            )
            if len(results) >= per_page or len(batch) < fetch_size:
                break
        return results[:per_page]

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers["Accept"] = "application/atom+xml, application/xml;q=0.9, */*;q=0.8"
        headers["User-Agent"] = str(self.config.user_agent or _DEFAULT_ARXIV_USER_AGENT)
        return headers

    def _build_params(
        self,
        *,
        query: str,
        search_prefix: str,
        start: int,
        per_page: int,
    ) -> dict[str, str]:
        return {
            "search_query": f"{search_prefix}:{query}",
            "start": str(max(0, int(start))),
            "max_results": str(per_page),
        }

    def _parse_feed(self, content: bytes) -> list[SearchProviderResult]:
        soup = BeautifulSoup(content, "xml")
        results: list[SearchProviderResult] = []
        for entry in soup.find_all("entry"):
            if not isinstance(entry, Tag):
                continue
            item = self._parse_entry(entry)
            if item is not None:
                results.append(item)
        return results

    def _parse_entry(self, entry: Tag) -> SearchProviderResult | None:
        title = self._text(self._find_first(entry, "title"))
        url = self._text(self._find_first(entry, "id"))
        if not title or not url:
            return None
        summary = self._text(self._find_first(entry, "summary"))
        authors = [
            author
            for author in (
                self._text(self._find_first(node, "name"))
                for node in entry.find_all("author")
            )
            if author
        ]
        return SearchProviderResult(
            url=url,
            title=title,
            snippet=self._build_snippet(summary=summary, authors=authors),
            engine=self.config.name,
            published_date=self._parse_published(
                self._text(self._find_first(entry, "published"))
            ),
        )

    def _build_snippet(self, *, summary: str, authors: list[str]) -> str:
        parts: list[str] = []
        if summary:
            parts.append(summary)
        if authors:
            parts.append(f"Authors: {', '.join(authors[:4])}")
        return " / ".join(parts)

    def _find_first(self, node: BeautifulSoup | Tag, *names: str) -> Tag | None:
        for name in names:
            found = node.find(name)
            if isinstance(found, Tag):
                return found
        return None

    def _text(self, node: Tag | None) -> str:
        if node is None:
            return ""
        return clean_whitespace(node.get_text(" ", strip=True))

    def _parse_published(self, value: str) -> str:
        token = clean_whitespace(value)
        if not token:
            return ""
        try:
            return normalize_iso8601_string(token)
        except ValueError:
            return ""

    def _resolve_search_prefix(self, value: Any) -> str:
        prefix = clean_whitespace(str(value or self.config.search_prefix or ""))
        return prefix or "all"

    def _coerce_per_page(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(_MAX_ARXIV_RESULTS_PER_PAGE, size))

    def _coerce_start(self, value: Any) -> int:
        try:
            start = int(value)
        except Exception:
            return 0
        return max(0, start)


__all__ = ["ArxivProvider", "ArxivProviderConfig"]
