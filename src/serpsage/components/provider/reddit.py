from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from typing_extensions import override
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import field_validator

from serpsage.components.http.base import HttpClientBase
from serpsage.components.provider.base import ProviderConfigBase, SearchProviderBase
from serpsage.dependencies import Depends
from serpsage.models.components.provider import (
    SearchProviderResponse,
    SearchProviderResult,
)
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


class RedditProvider(SearchProviderBase[RedditProviderConfig]):
    http: HttpClientBase = Depends()

    @override
    async def asearch(
        self,
        *,
        query: str,
        page: int = 1,
        language: str = "",
        **kwargs: Any,
    ) -> SearchProviderResponse:
        cfg = self.config
        normalized_query = clean_whitespace(query)
        normalized_language = clean_whitespace(language)
        if not normalized_query:
            raise ValueError("query must not be empty")

        extra = dict(kwargs)
        after = clean_whitespace(str(extra.pop("after", "") or ""))
        if int(page) > 1 and not after:
            return SearchProviderResponse(
                provider_backend="reddit",
                query=normalized_query,
                page=int(page),
                language=normalized_language,
                results=[],
                metadata={
                    "request_url": str(cfg.base_url),
                    "reddit_domain": str(cfg.base_url),
                    "skip_reason": "reddit pagination requires an `after` token",
                },
            )

        per_page = self._coerce_per_page(
            extra.pop("limit", extra.pop("per_page", cfg.results_per_page))
        )
        params = self._build_params(
            query=normalized_query,
            per_page=per_page,
            after=after,
            extra=extra,
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
        total_results = (
            int(listing["dist"])
            if isinstance(listing, dict)
            and isinstance(listing.get("dist"), (int, float))
            else None
        )
        metadata: dict[str, Any] = {
            "request_url": str(resp.url),
            "reddit_domain": str(cfg.base_url),
        }
        if isinstance(listing, dict):
            after_token = clean_whitespace(str(listing.get("after") or ""))
            before_token = clean_whitespace(str(listing.get("before") or ""))
            if after_token:
                metadata["after"] = after_token
            if before_token:
                metadata["before"] = before_token
        return SearchProviderResponse(
            provider_backend="reddit",
            query=normalized_query,
            page=int(page),
            language=normalized_language,
            total_results=total_results,
            results=results,
            metadata=metadata,
        )

    def _build_params(
        self,
        *,
        query: str,
        per_page: int,
        after: str,
        extra: dict[str, Any],
    ) -> dict[str, str]:
        params: dict[str, str] = {
            "q": query,
            "limit": str(per_page),
        }
        if after:
            params["after"] = after
        for key, value in extra.items():
            token = clean_whitespace(str(key or ""))
            if not token or value is None:
                continue
            params[token] = str(value)
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
            item = self._build_result(data)
            if item is None:
                continue
            key = item.url.casefold()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            if bool(item.metadata.get("thumbnail")):
                image_results.append(item)
            else:
                text_results.append(item)
        ordered = image_results + text_results
        for index, item in enumerate(ordered, start=1):
            item.position = index
        return ordered

    def _build_result(self, data: dict[str, Any]) -> SearchProviderResult | None:
        permalink = clean_whitespace(str(data.get("permalink") or ""))
        title = clean_whitespace(str(data.get("title") or ""))
        if not permalink or not title:
            return None
        url = urljoin("https://www.reddit.com/", permalink)
        metadata: dict[str, Any] = {}
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
        if thumbnail:
            metadata["thumbnail"] = thumbnail
        if external_url:
            metadata["external_url"] = external_url
        if created:
            metadata["published_date"] = created
        if subreddit:
            metadata["subreddit"] = subreddit
        if author:
            metadata["author"] = author
        if domain:
            metadata["domain"] = domain
        score = data.get("score")
        if isinstance(score, (int, float)):
            metadata["popularity"] = int(score)
        comments = data.get("num_comments")
        if isinstance(comments, (int, float)):
            metadata["comments_count"] = int(comments)
        over_18 = data.get("over_18")
        if isinstance(over_18, bool):
            metadata["nsfw"] = over_18
        return SearchProviderResult(
            url=url,
            title=title,
            snippet=" / ".join(snippet_parts),
            display_url=self._display_url(url),
            source_engine="reddit",
            metadata=metadata,
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
