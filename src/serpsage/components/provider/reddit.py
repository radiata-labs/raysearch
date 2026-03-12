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
        locale: str = "",
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        cfg = self.config
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        per_page = self._coerce_per_page(
            limit if limit is not None else cfg.results_per_page
        )
        after = clean_whitespace(str(kwargs.get("after") or ""))
        params = self._build_params(
            query=normalized_query,
            per_page=per_page,
            after=after,
            sort=kwargs.get("sort"),
            time_range=kwargs.get("time_range"),
        )
        headers = dict(cfg.headers or {})
        headers["Accept"] = "application/json"
        headers["User-Agent"] = str(cfg.user_agent or _DEFAULT_REDDIT_USER_AGENT)
        resp = await self.http.client.get(
            str(cfg.base_url),
            params=params,
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s),
            follow_redirects=bool(cfg.allow_redirects),
        )
        resp.raise_for_status()
        payload = resp.json()
        listing = payload.get("data") if isinstance(payload, dict) else {}
        raw_posts = listing.get("children") if isinstance(listing, dict) else []
        results = self._parse_results(raw_posts if isinstance(raw_posts, list) else [])
        _ = (
            int(listing["dist"])
            if isinstance(listing, dict)
            and isinstance(listing.get("dist"), (int, float))
            else None,
            str(resp.url),
            str(cfg.base_url),
            clean_whitespace(str(listing.get("after") or ""))
            if isinstance(listing, dict)
            else "",
            clean_whitespace(str(listing.get("before") or ""))
            if isinstance(listing, dict)
            else "",
            locale,
        )
        return results[:per_page]

    def _build_params(
        self,
        *,
        query: str,
        per_page: int,
        after: str,
        sort: Any,
        time_range: Any,
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
        normalized_time_range = clean_whitespace(str(time_range or "")).casefold()
        if normalized_time_range in {"hour", "day", "week", "month", "year", "all"}:
            params["t"] = normalized_time_range
        return params

    def _parse_results(
        self, raw_posts: list[dict[str, Any]]
    ) -> list[SearchProviderResult]:
        image_results: list[SearchProviderResult] = []
        text_results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()
        for wrapper in raw_posts:
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
        self, data: dict[str, Any]
    ) -> tuple[SearchProviderResult | None, bool]:
        permalink = clean_whitespace(str(data.get("permalink") or ""))
        title = clean_whitespace(str(data.get("title") or ""))
        if not permalink or not title:
            return None, False
        url = urljoin("https://www.reddit.com/", permalink)
        thumbnail = self._valid_thumbnail(data.get("thumbnail"))
        created = self._parse_created_utc(data.get("created_utc"))
        selftext = self._truncate_text(
            clean_whitespace(str(data.get("selftext") or ""))
        )
        external_url = clean_whitespace(str(data.get("url") or ""))
        subreddit = clean_whitespace(str(data.get("subreddit_name_prefixed") or ""))
        author = clean_whitespace(str(data.get("author") or ""))
        domain = clean_whitespace(str(data.get("domain") or ""))
        snippet_parts = [part for part in (subreddit, selftext) if part]
        if not snippet_parts and domain:
            snippet_parts.append(domain)
        if not snippet_parts and external_url and external_url != url:
            snippet_parts.append(external_url)
        _ = (
            thumbnail,
            external_url,
            created,
            author,
            domain,
            data.get("score"),
            data.get("num_comments"),
            data.get("over_18"),
            self._display_url(url),
        )
        return SearchProviderResult(
            url=url,
            title=title,
            snippet=" / ".join(snippet_parts),
            engine=self.config.name,
        ), bool(thumbnail)

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

    def _display_url(self, url: str) -> str:
        parsed = urlparse(url)
        host = clean_whitespace(parsed.netloc)
        path = clean_whitespace(parsed.path)
        return clean_whitespace(f"{host}{path}")

    def _coerce_per_page(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(_MAX_REDDIT_RESULTS_PER_PAGE, size))


__all__ = ["RedditProvider", "RedditProviderConfig"]
