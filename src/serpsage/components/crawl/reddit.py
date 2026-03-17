"""Reddit specialized crawler for fetching JSON data from Reddit's .json endpoint.

This crawler converts Reddit post URLs to their .json endpoint equivalent
to get structured data including comments, avoiding the need for HTML parsing
and Shadow DOM issues.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import override
from urllib.parse import urlparse

import httpx

from serpsage.components.crawl.base import CrawlerConfigBase, SpecializedCrawlerBase
from serpsage.components.http.base import HttpClientBase
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.dependencies import Depends
from serpsage.models.components.crawl import CrawlResult

# Reddit domains
_REDDIT_DOMAINS: frozenset[str] = frozenset(
    {"reddit.com", "www.reddit.com", "old.reddit.com", "new.reddit.com"}
)


class RedditCrawlerConfig(CrawlerConfigBase):
    __setting_family__ = "crawl"
    __setting_name__ = "reddit"

    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


class RedditCrawler(SpecializedCrawlerBase[RedditCrawlerConfig]):
    """Specialized crawler for Reddit that fetches JSON data.

    Converts Reddit post URLs to their .json endpoint to get structured data
    including comments, avoiding HTML parsing and Shadow DOM issues.
    """

    rate_limiter: RateLimiterBase[Any] = Depends()
    http: HttpClientBase = Depends()

    @classmethod
    @override
    def can_handle(cls, *, url: str) -> bool:
        """Return True if this crawler should handle the given URL."""
        return cls.is_reddit_url(url)

    @classmethod
    def is_reddit_url(cls, url: str) -> bool:
        """Check if URL is a Reddit post URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().removeprefix("www.")
            if domain not in _REDDIT_DOMAINS:
                return False
            path = parsed.path.lower()
            # Only handle post/comment URLs
            if "/comments/" not in path:
                return False
            # Avoid user profile pages
            return not ("/user/" in path or "/u/" in path)
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def to_json_url(cls, url: str) -> str:
        """Convert Reddit post URL to .json endpoint."""
        parsed = urlparse(url)
        json_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}.json"
        if parsed.query:
            json_url += f"?{parsed.query}"
        return json_url

    @override
    async def _acrawl(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> CrawlResult:
        """Fetch Reddit JSON endpoint."""
        if not self.is_reddit_url(url):
            raise ValueError(f"Not a Reddit post URL: {url}")

        json_url = self.to_json_url(url)
        host = urlparse(url).netloc.lower()

        await self.rate_limiter.acquire(host=host)
        try:
            resp = await self.http.client.get(
                json_url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": self.config.user_agent,
                },
                timeout=httpx.Timeout(self._resolve_timeout_s(timeout_s)),
                follow_redirects=True,
            )

            if resp.status_code == 200:
                return CrawlResult(
                    url=url,
                    status_code=resp.status_code,
                    content_type="application/json",
                    content=resp.content,
                    crawl_backend="reddit",
                    rendered=False,
                    content_kind="json",
                    headers=dict(resp.headers),
                    attempt_chain=["reddit:json_endpoint"],
                )

            # Return error result
            return CrawlResult(
                url=url,
                status_code=resp.status_code,
                content_type=resp.headers.get("content-type", ""),
                content=b"",
                crawl_backend="reddit",
                rendered=False,
                content_kind="unknown",
                headers=dict(resp.headers),
                attempt_chain=[f"reddit:error:{resp.status_code}"],
            )
        finally:
            await self.rate_limiter.release(host=host)

    def _resolve_timeout_s(self, timeout_s: float | None) -> float:
        return max(
            0.05, timeout_s if timeout_s and timeout_s > 0 else self.config.timeout_s
        )


__all__ = ["RedditCrawler", "RedditCrawlerConfig"]
