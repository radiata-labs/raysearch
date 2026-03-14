"""Reddit search provider backed by the public JSON listing endpoint.

This provider follows the practical shape of a Reddit engine:

- it queries ``/search.json`` and paginates through ``after`` cursors
- it supports Reddit sort and coarse time-range controls
- it parses post metadata into compact text-oriented search results
- it can over-fetch and filter locally when date bounds are requested

Configuration
=============

Example configuration in this project:

.. code:: yaml

   reddit:
     enabled: true
     base_url: https://www.reddit.com/search.json
     allow_redirects: true
     user_agent: serpsage-reddit-provider/1.0
     results_per_page: 25

Notes
=====

- Reddit's public search is coarse and ranking-heavy, so the provider may scan
  several pages when date bounds are supplied.
- The provider emits the post permalink rather than an external outbound URL to
  keep fetch behavior stable.
- Thumbnail information is used only to bias ordering within a page, not as
  part of the result schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from typing_extensions import override
from urllib.parse import urljoin, urlparse

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
from serpsage.utils import clean_whitespace

_DEFAULT_REDDIT_BASE_URL = "https://www.reddit.com/search.json"
_DEFAULT_REDDIT_USER_AGENT = "serpsage-reddit-provider/1.0"
_DEFAULT_REDDIT_RESULTS_PER_PAGE = 25
_MAX_REDDIT_RESULTS_PER_PAGE = 100
_MAX_REDDIT_DATE_SCAN_PAGES = 5
_REDDIT_INVALID_THUMBNAILS = {"", "default", "self", "nsfw", "spoiler", "image"}


class RedditProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "reddit"

    base_url: str = _DEFAULT_REDDIT_BASE_URL
    allow_redirects: bool = True
    user_agent: str = _DEFAULT_REDDIT_USER_AGENT
    results_per_page: int = _DEFAULT_REDDIT_RESULTS_PER_PAGE

    @field_validator("user_agent")
    @classmethod
    def _normalize_user_agent(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("reddit results_per_page must be > 0")
        if size > _MAX_REDDIT_RESULTS_PER_PAGE:
            raise ValueError(
                f"reddit results_per_page must be <= {_MAX_REDDIT_RESULTS_PER_PAGE}"
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
        if env.get("REDDIT_BASE_URL"):
            payload["base_url"] = env["REDDIT_BASE_URL"]
        return payload


class RedditProvider(
    SearchProviderBase[RedditProviderConfig],
    meta=ProviderMeta(
        name="reddit",
        website="https://www.reddit.com/",
        description="Community discussion search across Reddit posts, niche forums, and user-generated recommendations.",
        preference="Prefer experience-driven queries, opinions, troubleshooting, comparisons, recommendations, and community discussion topics.",
        categories=["social media"],
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
        cfg = self.config
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        per_page = self._coerce_per_page(
            limit if limit is not None else cfg.results_per_page
        )
        headers = self._build_headers()
        time_range = self._resolve_time_range(
            runtime_value=kwargs.get("time_range"),
            start_published_date=start_published_date,
            end_published_date=end_published_date,
        )
        after = clean_whitespace(str(kwargs.get("after") or ""))
        should_backfill = bool(start_published_date or end_published_date)
        max_pages = _MAX_REDDIT_DATE_SCAN_PAGES if should_backfill else 1
        initial_fetch_size = (
            self._initial_dated_fetch_limit(
                target_limit=per_page,
                max_limit=_MAX_REDDIT_RESULTS_PER_PAGE,
            )
            if should_backfill
            else per_page
        )
        results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()

        for page_index in range(max_pages):
            fetch_size = initial_fetch_size if page_index == 0 else per_page
            resp = await self.http.client.get(
                str(cfg.base_url),
                params=self._build_params(
                    query=normalized_query,
                    per_page=fetch_size,
                    after=after,
                    sort=kwargs.get("sort"),
                    time_range=time_range,
                ),
                headers=headers,
                timeout=httpx.Timeout(cfg.timeout_s),
                follow_redirects=bool(cfg.allow_redirects),
            )
            resp.raise_for_status()
            payload = resp.json()
            listing = payload.get("data") if isinstance(payload, dict) else None
            batch = self._parse_results(
                listing.get("children") if isinstance(listing, dict) else []
            )
            batch = self._filter_results_by_published_date(
                results=batch,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
            )
            for item in batch:
                key = item.url.casefold()
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                results.append(item)
            next_after = (
                clean_whitespace(str(listing.get("after") or ""))
                if isinstance(listing, dict)
                else ""
            )
            raw_dist = listing.get("dist") if isinstance(listing, dict) else None
            listing_count = (
                int(raw_dist) if isinstance(raw_dist, (int, float)) else len(batch)
            )
            if (
                len(results) >= per_page
                or not next_after
                or next_after == after
                or listing_count < fetch_size
            ):
                break
            after = next_after
        return results[:per_page]

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers["Accept"] = "application/json"
        headers["User-Agent"] = str(
            self.config.user_agent or _DEFAULT_REDDIT_USER_AGENT
        )
        return headers

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

    def _build_params(
        self,
        *,
        query: str,
        per_page: int,
        after: str,
        sort: Any,
        time_range: str,
    ) -> dict[str, str]:
        params: dict[str, str] = {
            "q": query,
            "limit": str(per_page),
        }
        if after:
            params["after"] = after
        normalized_sort = clean_whitespace(str(sort or "")).casefold()
        if normalized_sort in {"relevance", "hot", "top", "new", "comments"}:
            params["sort"] = normalized_sort
        if time_range in {"hour", "day", "week", "month", "year", "all"}:
            params["t"] = time_range
        return params

    def _parse_results(self, raw_posts: Any) -> list[SearchProviderResult]:
        image_results: list[SearchProviderResult] = []
        text_results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()
        for wrapper in raw_posts if isinstance(raw_posts, list) else []:
            if not isinstance(wrapper, dict):
                continue
            data = wrapper.get("data")
            if not isinstance(data, dict):
                continue
            item, has_thumbnail = self._build_result(data)
            if item is None:
                continue
            key = item.url.casefold()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            if has_thumbnail:
                image_results.append(item)
            else:
                text_results.append(item)
        return image_results + text_results

    def _build_result(
        self,
        data: dict[str, Any],
    ) -> tuple[SearchProviderResult | None, bool]:
        permalink = clean_whitespace(str(data.get("permalink") or ""))
        title = clean_whitespace(str(data.get("title") or ""))
        if not permalink or not title:
            return None, False
        url = urljoin("https://www.reddit.com/", permalink)
        selftext = self._truncate_text(
            clean_whitespace(str(data.get("selftext") or ""))
        )
        subreddit = clean_whitespace(str(data.get("subreddit_name_prefixed") or ""))
        domain = clean_whitespace(str(data.get("domain") or ""))
        external_url = clean_whitespace(str(data.get("url") or ""))
        snippet_parts = [part for part in (subreddit, selftext) if part]
        if not snippet_parts and domain:
            snippet_parts.append(domain)
        if not snippet_parts and external_url and external_url != url:
            snippet_parts.append(external_url)
        return (
            SearchProviderResult(
                url=url,
                title=title,
                snippet=" / ".join(snippet_parts),
                engine=self.config.name,
                published_date=self._parse_created_utc(data.get("created_utc")),
            ),
            bool(self._valid_thumbnail(data.get("thumbnail"))),
        )

    def _valid_thumbnail(self, value: Any) -> str:
        thumbnail = clean_whitespace(str(value or ""))
        if not thumbnail or thumbnail.casefold() in _REDDIT_INVALID_THUMBNAILS:
            return ""
        parsed = urlparse(thumbnail)
        if parsed.scheme not in {"http", "https"}:
            return ""
        if not parsed.netloc or not parsed.path:
            return ""
        return thumbnail

    def _parse_created_utc(self, value: Any) -> str:
        try:
            timestamp = float(value)
        except Exception:
            return ""
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()

    def _truncate_text(self, value: str, *, limit: int = 500) -> str:
        text = clean_whitespace(value)
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    def _coerce_per_page(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(_MAX_REDDIT_RESULTS_PER_PAGE, size))


__all__ = ["RedditProvider", "RedditProviderConfig"]
