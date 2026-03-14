# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OpenAlex engine integrates the OpenAlex Works API.

This provider follows the same broad shape as the SearXNG engine:

- it uses the official Works endpoint and does not require an API key
- it searches through the ``search`` parameter and keeps relevance sorting
- it can apply a language filter derived from the runtime locale
- it reconstructs abstracts from OpenAlex's inverted index payload
- it carries over authors, venue, tags, and citation hints into the snippet

Configuration
=============

Example configuration in this project:

.. code:: yaml

   openalex:
     enabled: true
     base_url: https://api.openalex.org/works
     user_agent: serpsage-openalex-provider/1.0
     results_per_page: 10
     mailto: ""

Notes
=====

- ``mailto`` is optional but recommended by OpenAlex for polite-pool access.
- OpenAlex may return ``publication_date`` in ``YYYY``, ``YYYY-MM``, or
  ``YYYY-MM-DD`` form, so this provider normalizes all of them before storing
  ``published_date``.
- The primary landing page is preferred as the result URL, with the work ID as a
  fallback when no landing page is available.

Implementation
==============

The provider keeps the request shape close to the upstream OpenAlex query model
and folds the richest metadata fields into the emitted snippet.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import httpx
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

_DEFAULT_OPENALEX_BASE_URL = "https://api.openalex.org/works"
_DEFAULT_OPENALEX_USER_AGENT = "serpsage-openalex-provider/1.0"
_DEFAULT_OPENALEX_RESULTS_PER_PAGE = 10
_MAX_OPENALEX_RESULTS_PER_PAGE = 200
_MAX_OPENALEX_SCAN_PAGES = 5


class OpenAlexProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "openalex"

    base_url: str = _DEFAULT_OPENALEX_BASE_URL
    user_agent: str = _DEFAULT_OPENALEX_USER_AGENT
    results_per_page: int = _DEFAULT_OPENALEX_RESULTS_PER_PAGE
    mailto: str = ""

    @field_validator("user_agent", "mailto")
    @classmethod
    def _normalize_text_fields(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("openalex results_per_page must be > 0")
        if size > _MAX_OPENALEX_RESULTS_PER_PAGE:
            raise ValueError(
                f"openalex results_per_page must be <= {_MAX_OPENALEX_RESULTS_PER_PAGE}"
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
        if env.get("OPENALEX_BASE_URL"):
            payload["base_url"] = env["OPENALEX_BASE_URL"]
        if env.get("OPENALEX_MAILTO"):
            payload["mailto"] = env["OPENALEX_MAILTO"]
        return payload


class OpenAlexProvider(
    SearchProviderBase[OpenAlexProviderConfig],
    meta=ProviderMeta(
        name="openalex",
        website="https://openalex.org/",
        description="Scientific paper search through the OpenAlex Works API with broad research coverage beyond preprints.",
        preference="Prefer academic topic, author, method, venue, and literature-discovery queries where broad scholarly coverage matters.",
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
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        per_page = self._coerce_per_page(
            limit if limit is not None else self.config.results_per_page
        )
        should_backfill = bool(start_published_date or end_published_date)
        max_pages = _MAX_OPENALEX_SCAN_PAGES if should_backfill else 1
        page = self._coerce_page(kwargs.get("page"))
        request_size = (
            self._initial_dated_fetch_limit(
                target_limit=per_page,
                max_limit=_MAX_OPENALEX_RESULTS_PER_PAGE,
            )
            if should_backfill
            else per_page
        )
        results: list[SearchProviderResult] = []

        for page_index in range(max_pages):
            resp = await self.http.client.get(
                str(self.config.base_url),
                params=self._build_params(
                    query=normalized_query,
                    page=page + page_index,
                    per_page=request_size if page_index == 0 else per_page,
                    locale=locale,
                ),
                headers=self._build_headers(),
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

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers.setdefault("Accept", "application/json")
        headers["User-Agent"] = (
            clean_whitespace(str(self.config.user_agent or ""))
            or _DEFAULT_OPENALEX_USER_AGENT
        )
        return headers

    def _build_params(
        self,
        *,
        query: str,
        page: int,
        per_page: int,
        locale: str,
    ) -> dict[str, str]:
        params = {
            "search": query,
            "page": str(max(1, int(page))),
            "per-page": str(max(1, int(per_page))),
            "sort": "relevance_score:desc",
        }
        filters = self._build_filters(locale=locale)
        if filters:
            params["filter"] = ",".join(filters)
        if self.config.mailto:
            params["mailto"] = self.config.mailto
        return params

    def _build_filters(self, *, locale: str) -> list[str]:
        normalized = clean_whitespace(locale).replace("_", "-")
        if not normalized or normalized.casefold() == "all":
            return []
        language = clean_whitespace(normalized.split("-", 1)[0]).lower()
        if len(language) == 2:
            return [f"language:{language}"]
        return []

    def _parse_results(self, raw_results: Any) -> list[SearchProviderResult]:
        results: list[SearchProviderResult] = []
        for item in raw_results if isinstance(raw_results, list) else []:
            if not isinstance(item, dict):
                continue
            url = self._extract_url(item)
            if not url:
                continue
            title = clean_whitespace(str(item.get("title") or ""))
            if not title:
                continue
            results.append(
                SearchProviderResult(
                    url=url,
                    title=title,
                    snippet=self._build_snippet(item),
                    engine=self.config.name,
                    published_date=self._parse_publication_date(
                        item.get("publication_date")
                    ),
                )
            )
        return results

    def _extract_url(self, item: dict[str, Any]) -> str:
        primary_location = item.get("primary_location")
        if isinstance(primary_location, dict):
            landing_page_url = clean_whitespace(
                str(primary_location.get("landing_page_url") or "")
            )
            if landing_page_url:
                return landing_page_url
        return clean_whitespace(str(item.get("id") or ""))

    def _build_snippet(self, item: dict[str, Any]) -> str:
        abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))
        authors = self._extract_authors(item)
        venue = self._extract_venue(item)
        tags = self._extract_tags(item)
        comments = self._extract_comments(item)
        parts: list[str] = []
        if abstract:
            parts.append(abstract)
        if authors:
            parts.append(f"Authors: {', '.join(authors[:4])}")
        if venue:
            parts.append(venue)
        if tags:
            parts.append(f"Topics: {', '.join(tags[:4])}")
        if comments:
            parts.append(comments)
        return clean_whitespace(" / ".join(parts))

    def _reconstruct_abstract(self, raw: Any) -> str:
        if not isinstance(raw, dict) or not raw:
            return ""
        position_to_token: dict[int, str] = {}
        max_index = -1
        for token, positions in raw.items():
            clean_token = clean_whitespace(str(token or ""))
            if not clean_token or not isinstance(positions, list):
                continue
            for position in positions:
                try:
                    index = int(position)
                except (TypeError, ValueError):
                    continue
                position_to_token[index] = clean_token
                max_index = max(max_index, index)
        if max_index < 0:
            return ""
        return clean_whitespace(
            " ".join(position_to_token.get(index, "") for index in range(max_index + 1))
        )

    def _extract_authors(self, item: dict[str, Any]) -> list[str]:
        authors: list[str] = []
        for raw_authorship in item.get("authorships", []):
            if not isinstance(raw_authorship, dict):
                continue
            author = raw_authorship.get("author")
            if not isinstance(author, dict):
                continue
            name = clean_whitespace(str(author.get("display_name") or ""))
            if name:
                authors.append(name)
        return authors

    def _extract_venue(self, item: dict[str, Any]) -> str:
        primary_location = item.get("primary_location")
        source = (
            primary_location.get("source")
            if isinstance(primary_location, dict)
            else None
        )
        if isinstance(source, dict):
            venue = clean_whitespace(str(source.get("display_name") or ""))
            if venue:
                return venue
        host_venue = item.get("host_venue")
        if isinstance(host_venue, dict):
            venue = clean_whitespace(str(host_venue.get("display_name") or ""))
            if venue:
                return venue
        return ""

    def _extract_tags(self, item: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for raw_concept in item.get("concepts", []):
            if not isinstance(raw_concept, dict):
                continue
            name = clean_whitespace(str(raw_concept.get("display_name") or ""))
            if name:
                tags.append(name)
        return tags

    def _extract_comments(self, item: dict[str, Any]) -> str:
        cited_by_count = item.get("cited_by_count")
        if isinstance(cited_by_count, int):
            return f"{cited_by_count} citations"
        return ""

    def _parse_publication_date(self, value: Any) -> str:
        token = clean_whitespace(str(value or ""))
        if not token:
            return ""
        if len(token) == 4 and token.isdigit():
            token = f"{token}-01-01"
        elif len(token) == 7 and token.count("-") == 1:
            token = f"{token}-01"
        try:
            return normalize_iso8601_string(token)
        except ValueError:
            return ""

    def _coerce_per_page(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(_MAX_OPENALEX_RESULTS_PER_PAGE, size))

    def _coerce_page(self, value: Any) -> int:
        try:
            page = int(value)
        except Exception:
            return 1
        return max(1, page)


__all__ = ["OpenAlexProvider", "OpenAlexProviderConfig"]
