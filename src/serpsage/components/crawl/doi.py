"""DOI specialized crawler for fetching metadata from Crossref API.

This crawler extracts DOI identifiers from URLs and fetches structured metadata
from the Crossref API, providing rich bibliographic information including:
- Title, authors, and abstract
- Journal/venue information
- Publication date
- Citation count
- References and citations

Supported URL patterns:
- https://doi.org/10.1234/example
- https://dx.doi.org/10.1234/example
- Direct DOI strings: 10.1234/example
"""

from __future__ import annotations

import re
from typing import Any
from typing_extensions import override
from urllib.parse import unquote, urlparse

import httpx

from serpsage.components.crawl.base import CrawlerConfigBase, SpecializedCrawlerBase
from serpsage.components.http.base import HttpClientBase
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.dependencies import Depends
from serpsage.models.components.crawl import CrawlResult

# DOI pattern: 10.XXXX/XXXXX
_DOI_PATTERN = re.compile(r"10\.\d{4,}/[^\s<>\"{}|\\^`\[\]]+", re.IGNORECASE)

# DOI URL domains
_DOI_URL_DOMAINS: frozenset[str] = frozenset({"doi.org", "www.doi.org", "dx.doi.org"})

# Crossref API base URL
_CROSSREF_API_BASE = "https://api.crossref.org/works/"
_CROSSREF_USER_AGENT = "serpsage-doi-crawler/1.0 (mailto:contact@example.com)"


class DOICrawlerConfig(CrawlerConfigBase):
    __setting_family__ = "crawl"
    __setting_name__ = "doi"

    user_agent: str = _CROSSREF_USER_AGENT
    mailto: str = ""  # Optional email for Crossref polite pool


class DOICrawler(SpecializedCrawlerBase[DOICrawlerConfig]):
    """Specialized crawler for DOI metadata via Crossref API.

    Extracts DOI identifiers from various URL formats and fetches rich
    bibliographic metadata from the Crossref API.
    """

    rate_limiter: RateLimiterBase[Any] = Depends()
    http: HttpClientBase = Depends()

    @classmethod
    @override
    def can_handle(cls, *, url: str) -> bool:
        """Return True if this crawler should handle the given URL."""
        return cls.is_doi_url(url)

    @classmethod
    def extract_doi(cls, url_or_doi: str) -> str | None:
        """Extract DOI from URL or return DOI string if valid.

        Args:
            url_or_doi: URL containing DOI or direct DOI string

        Returns:
            Normalized DOI string (lowercase) or None if not found
        """
        text = url_or_doi.strip()

        # Try to parse as URL first
        try:
            parsed = urlparse(text)
            domain = parsed.netloc.lower().removeprefix("www.")

            if domain in _DOI_URL_DOMAINS:
                # Extract DOI from path
                path = unquote(parsed.path).strip("/")
                if path:
                    # Validate it looks like a DOI
                    match = _DOI_PATTERN.search(path)
                    if match:
                        return match.group(0).lower()
        except Exception:
            pass

        # Try direct DOI match
        match = _DOI_PATTERN.search(text)
        if match:
            return match.group(0).lower()

        return None

    @classmethod
    def is_doi_url(cls, url: str) -> bool:
        """Check if URL is a DOI URL or a valid DOI string."""
        return cls.extract_doi(url) is not None

    @classmethod
    def to_crossref_url(cls, doi: str) -> str:
        """Convert DOI to Crossref API URL."""
        # Ensure DOI is URL-safe
        encoded_doi = httpx.URL(doi).path.lstrip("/")
        return f"{_CROSSREF_API_BASE}{encoded_doi}"

    @override
    async def _acrawl(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> CrawlResult:
        """Fetch DOI metadata from Crossref API."""
        doi = self.extract_doi(url)
        if not doi:
            raise ValueError(f"Not a valid DOI: {url}")

        crossref_url = self.to_crossref_url(doi)

        # Build headers with optional mailto for polite pool
        headers = self._build_headers()

        await self.rate_limiter.acquire(host="api.crossref.org")
        try:
            resp = await self.http.client.get(
                crossref_url,
                headers=headers,
                timeout=httpx.Timeout(self._resolve_timeout_s(timeout_s)),
                follow_redirects=True,
            )

            if resp.status_code == 200:
                return CrawlResult(
                    url=url,
                    status_code=resp.status_code,
                    content_type="application/json",
                    content=resp.content,
                    crawl_backend="doi",
                    rendered=False,
                    content_kind="json",
                    headers=dict(resp.headers),
                    attempt_chain=["doi:crossref_api"],
                )

            # Return error result
            return CrawlResult(
                url=url,
                status_code=resp.status_code,
                content_type=resp.headers.get("content-type", ""),
                content=b"",
                crawl_backend="doi",
                rendered=False,
                content_kind="unknown",
                headers=dict(resp.headers),
                attempt_chain=[f"doi:error:{resp.status_code}"],
            )
        finally:
            await self.rate_limiter.release(host="api.crossref.org")

    def _build_headers(self) -> dict[str, str]:
        """Build request headers for Crossref API."""
        headers = {
            "Accept": "application/json",
            "User-Agent": self.config.user_agent,
        }
        # Add mailto for Crossref polite pool (better rate limits)
        if self.config.mailto:
            # Include mailto in User-Agent for polite pool
            headers["User-Agent"] = (
                f"{self.config.user_agent.split('(')[0].strip()} (mailto:{self.config.mailto})"
            )
        return headers

    def _resolve_timeout_s(self, timeout_s: float | None) -> float:
        return max(
            0.05, timeout_s if timeout_s and timeout_s > 0 else self.config.timeout_s
        )


__all__ = ["DOICrawler", "DOICrawlerConfig"]
