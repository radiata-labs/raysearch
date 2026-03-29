# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reuters is an international news agency with a searchable article feed.

This provider keeps the core request pattern used by the SearXNG Reuters
engine:

- it queries Reuters' ``articles-by-search-v2`` endpoint
- it encodes the request body as a JSON string inside the ``query`` parameter
- it supports Reuters' native relevance and display-date sort modes
- it applies Reuters' start-date filtering when a time range is requested

Configuration
=============

Example configuration in this project:

.. code:: yaml

   reuters:
     enabled: true
     base_url: https://www.reuters.com
     user_agent: raysearch-reuters-provider/1.0
     results_per_page: 20
     sort_order: relevance

Notes
=====

- ``sort_order`` supports Reuters' native ordering values such as
  ``relevance`` and ``display_date:desc``.
- Reuters' search endpoint accepts an encoded JSON payload in the ``query``
  parameter rather than ordinary flat query-string fields.
- The provider normalizes ``display_time`` into ``published_date`` so later
  date filters and ranking stages can consume it consistently.

Implementation
==============

The provider keeps Reuters-specific request encoding isolated so the rest of the
pipeline still sees ordinary normalized ``SearchProviderResult`` objects.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from json import dumps
from typing import Any
from typing_extensions import override
from urllib.parse import quote_plus, urljoin

import httpx
from pydantic import field_validator

from raysearch.components.http.base import HttpClientBase
from raysearch.components.provider.base import (
    ProviderConfigBase,
    ProviderMeta,
    SearchProviderBase,
)
from raysearch.dependencies import Depends
from raysearch.models.components.provider import SearchProviderResult
from raysearch.utils import (
    clean_whitespace,
    normalize_iso8601_string,
    parse_iso8601_datetime,
)

_DEFAULT_REUTERS_BASE_URL = "https://www.reuters.com"
_DEFAULT_REUTERS_USER_AGENT = "raysearch-reuters-provider/1.0"
_DEFAULT_REUTERS_RESULTS_PER_PAGE = 20
_MAX_REUTERS_RESULTS_PER_PAGE = 50
_MAX_REUTERS_SCAN_PAGES = 5
_REUTERS_SEARCH_PATH = "/pf/api/v3/content/fetch/articles-by-search-v2"
_REUTERS_TIME_RANGE_DURATION_MAP = {
    "day": 1,
    "week": 7,
    "month": 30,
    "year": 365,
}


class ReutersProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "reuters"

    base_url: str = _DEFAULT_REUTERS_BASE_URL
    user_agent: str = _DEFAULT_REUTERS_USER_AGENT
    results_per_page: int = _DEFAULT_REUTERS_RESULTS_PER_PAGE
    sort_order: str = "relevance"

    @field_validator("user_agent", "sort_order")
    @classmethod
    def _normalize_text_fields(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("reuters results_per_page must be > 0")
        if size > _MAX_REUTERS_RESULTS_PER_PAGE:
            raise ValueError(
                f"reuters results_per_page must be <= {_MAX_REUTERS_RESULTS_PER_PAGE}"
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
        if env.get("REUTERS_BASE_URL"):
            payload["base_url"] = env["REUTERS_BASE_URL"]
        return payload


class ReutersProvider(
    SearchProviderBase[ReutersProviderConfig],
    meta=ProviderMeta(
        name="reuters",
        website="https://www.reuters.com/",
        description="Reuters news search with article metadata and date-aware filtering from Reuters' own search backend.",
        preference="Prefer business news, policy news, finance, legal developments, and current-events searches where primary newswire coverage matters.",
        categories=["news"],
    ),
):
    http: HttpClientBase = Depends()

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
        max_pages = _MAX_REUTERS_SCAN_PAGES if should_backfill else 1
        page = self._coerce_page(kwargs.get("page"))
        time_range = self._resolve_time_range(
            runtime_value=kwargs.get("time_range"),
            start_published_date=start_published_date,
            end_published_date=end_published_date,
        )
        results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()

        for page_index in range(max_pages):
            request_url = self._build_request_url(
                query=normalized_query,
                offset=(page + page_index - 1) * per_page,
                per_page=per_page,
                sort_order=kwargs.get("sort_order"),
                time_range=time_range,
                start_published_date=start_published_date,
            )
            resp = await self.http.client.get(
                request_url,
                headers=self._build_headers(),
                timeout=httpx.Timeout(self.config.timeout_s),
                follow_redirects=bool(self.config.allow_redirects),
            )
            resp.raise_for_status()
            batch = self._parse_results(resp.json())
            batch = self._filter_results_by_published_date(
                results=batch,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
            )
            if not batch:
                break
            for item in batch:
                key = item.url.casefold()
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                results.append(item)
            if len(results) >= per_page:
                break
        return results[:per_page]

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers.setdefault("Accept", "application/json")
        headers["User-Agent"] = (
            clean_whitespace(str(self.config.user_agent or ""))
            or _DEFAULT_REUTERS_USER_AGENT
        )
        return headers

    def _build_request_url(
        self,
        *,
        query: str,
        offset: int,
        per_page: int,
        sort_order: Any,
        time_range: str,
        start_published_date: str | None,
    ) -> str:
        payload: dict[str, Any] = {
            "keyword": query,
            "offset": max(0, int(offset)),
            "orderby": self._resolve_sort_order(sort_order),
            "size": max(1, int(per_page)),
            "website": "reuters",
        }
        start_date = self._resolve_start_date(
            time_range=time_range,
            start_published_date=start_published_date,
        )
        if start_date:
            payload["start_date"] = start_date
        encoded_query = quote_plus(dumps(payload, separators=(",", ":")))
        return (
            f"{self.config.base_url.rstrip('/')}{_REUTERS_SEARCH_PATH}"
            f"?query={encoded_query}"
        )

    def _resolve_sort_order(self, value: Any) -> str:
        token = clean_whitespace(str(value or self.config.sort_order or "")).casefold()
        if token in {"relevance", "display_date:desc", "display_date:asc"}:
            return token
        return "relevance"

    def _resolve_time_range(
        self,
        *,
        runtime_value: Any,
        start_published_date: str | None,
        end_published_date: str | None,
    ) -> str:
        token = clean_whitespace(str(runtime_value or "")).casefold()
        if token:
            return token
        return self._relative_time_range_from_bounds(
            start_published_date=start_published_date,
            end_published_date=end_published_date,
        )

    def _resolve_start_date(
        self,
        *,
        time_range: str,
        start_published_date: str | None,
    ) -> str:
        if time_range in _REUTERS_TIME_RANGE_DURATION_MAP:
            days = _REUTERS_TIME_RANGE_DURATION_MAP[time_range]
            return (datetime.now(tz=UTC) - timedelta(days=days)).isoformat()
        start_value = clean_whitespace(str(start_published_date or ""))
        if not start_value:
            return ""
        start_at = parse_iso8601_datetime(start_value)
        if start_at is None:
            return ""
        return start_at.isoformat()

    def _parse_results(self, payload: Any) -> list[SearchProviderResult]:
        result_block = payload.get("result") if isinstance(payload, dict) else None
        articles = (
            result_block.get("articles") if isinstance(result_block, dict) else []
        )
        results: list[SearchProviderResult] = []
        for article in articles if isinstance(articles, list) else []:
            if not isinstance(article, dict):
                continue
            canonical_url = clean_whitespace(str(article.get("canonical_url") or ""))
            title = clean_whitespace(str(article.get("web") or ""))
            if not canonical_url or not title:
                continue
            description = clean_whitespace(str(article.get("description") or ""))
            kicker = clean_whitespace(
                str(article.get("kicker", {}).get("name") or "")
                if isinstance(article.get("kicker"), dict)
                else ""
            )
            snippet = clean_whitespace(
                " / ".join(part for part in (kicker, description) if part)
            )
            results.append(
                SearchProviderResult(
                    url=urljoin(str(self.config.base_url), canonical_url),
                    title=title,
                    snippet=snippet,
                    engine=self.config.name,
                    published_date=self._parse_published_date(
                        article.get("display_time")
                    ),
                )
            )
        return results

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
        return max(1, min(_MAX_REUTERS_RESULTS_PER_PAGE, size))

    def _coerce_page(self, value: Any) -> int:
        try:
            page = int(value)
        except Exception:
            return 1
        return max(1, page)


__all__ = ["ReutersProvider", "ReutersProviderConfig"]
