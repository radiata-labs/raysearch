# SPDX-License-Identifier: AGPL-3.0-or-later
"""Marginalia Search is an independent open web search engine.

This adapter follows the structure of the corresponding SearXNG engine:

- it uses the official JSON API
- it requires an API key
- it sends the search query through the ``query`` parameter
- it forwards a lightweight NSFW control derived from safesearch

Configuration
=============

The provider requires an API key. Example configuration in this project:

.. code:: yaml

   marginalia:
     enabled: true
     base_url: https://api2.marginalia-search.com
     user_agent: serpsage-marginalia-provider/1.0
     api_key: "..."
     results_per_page: 20

Notes
=====

- The upstream API key is mandatory; without it this provider raises at runtime.
- The request is sent to ``/search`` with ``count``, ``nsfw``, and ``query``
  parameters, matching the successful SearXNG integration pattern.
- Result parsing keeps the title, URL, description, and auxiliary details, which
  is usually enough for ranking and later fetch stages.

Implementation
==============

The provider keeps the request and response handling deliberately narrow because
the upstream API is already structured and lightweight.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import override
from urllib.parse import urlencode

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

_DEFAULT_MARGINALIA_BASE_URL = "https://api2.marginalia-search.com"
_DEFAULT_MARGINALIA_USER_AGENT = "serpsage-marginalia-provider/1.0"
_DEFAULT_MARGINALIA_RESULTS_PER_PAGE = 20
_MAX_MARGINALIA_RESULTS_PER_PAGE = 100


class MarginaliaProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "marginalia"

    base_url: str = _DEFAULT_MARGINALIA_BASE_URL
    user_agent: str = _DEFAULT_MARGINALIA_USER_AGENT
    api_key: str | None = None
    results_per_page: int = _DEFAULT_MARGINALIA_RESULTS_PER_PAGE

    @field_validator("user_agent")
    @classmethod
    def _normalize_user_agent(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("api_key")
    @classmethod
    def _normalize_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        token = clean_whitespace(str(value))
        return token or None

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("marginalia results_per_page must be > 0")
        if size > _MAX_MARGINALIA_RESULTS_PER_PAGE:
            raise ValueError(
                f"marginalia results_per_page must be <= {_MAX_MARGINALIA_RESULTS_PER_PAGE}"
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
        if env.get("MARGINALIA_BASE_URL"):
            payload["base_url"] = env["MARGINALIA_BASE_URL"]
        if env.get("MARGINALIA_API_KEY"):
            payload["api_key"] = env["MARGINALIA_API_KEY"]
        return payload


class MarginaliaProvider(
    SearchProviderBase[MarginaliaProviderConfig],
    meta=ProviderMeta(
        name="marginalia",
        website="https://marginalia.nu/",
        description="Independent web search focused on the open web, small sites, and less SEO-heavy pages.",
        preference="Prefer personal sites, blogs, niche web pages, and independent websites that broad web search often under-ranks.",
        categories=["general", "web"],
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
        if not self.config.api_key:
            raise RuntimeError("marginalia provider requires an api_key")

        page_size = self._coerce_page_size(
            limit if limit is not None else self.config.results_per_page
        )
        resp = await self.http.client.get(
            self._build_request_url(
                query=normalized_query,
                page_size=page_size,
                safesearch=kwargs.get("safesearch"),
                moderation=moderation,
            ),
            headers=self._build_headers(),
            timeout=httpx.Timeout(self.config.timeout_s),
            follow_redirects=bool(self.config.allow_redirects),
        )
        resp.raise_for_status()
        return self._parse_results(resp.json().get("results"))

    def _build_request_url(
        self,
        *,
        query: str,
        page_size: int,
        safesearch: Any,
        moderation: bool,
    ) -> str:
        params = {
            "count": str(page_size),
            "nsfw": str(
                min(
                    self._resolve_safesearch_level(
                        safesearch,
                        moderation=moderation,
                    ),
                    1,
                )
            ),
            "query": query,
        }
        base_url = clean_whitespace(
            str(self.config.base_url or _DEFAULT_MARGINALIA_BASE_URL)
        )
        return f"{base_url.rstrip('/')}/search?{urlencode(params)}"

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers.setdefault("Accept", "application/json")
        headers["User-Agent"] = (
            clean_whitespace(str(self.config.user_agent or ""))
            or _DEFAULT_MARGINALIA_USER_AGENT
        )
        headers["API-Key"] = str(self.config.api_key)
        return headers

    def _parse_results(self, raw_results: Any) -> list[SearchProviderResult]:
        results: list[SearchProviderResult] = []
        for item in raw_results if isinstance(raw_results, list) else []:
            if not isinstance(item, dict):
                continue
            url = clean_whitespace(str(item.get("url") or ""))
            title = clean_whitespace(str(item.get("title") or ""))
            if not url or not title:
                continue
            snippet = clean_whitespace(str(item.get("description") or ""))
            details = clean_whitespace(str(item.get("details") or ""))
            if details:
                snippet = clean_whitespace(
                    " / ".join(part for part in (snippet, details) if part)
                )
            results.append(
                SearchProviderResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    engine=self.config.name,
                )
            )
        return results

    def _resolve_safesearch_level(self, value: Any, *, moderation: bool) -> int:
        token = clean_whitespace(str(value or "")).casefold()
        if token in {"2", "high", "strict"}:
            return 2
        if token in {"1", "medium", "moderate"}:
            return 1
        if token in {"0", "off", "false", "no"}:
            return 0
        return 1 if moderation else 0

    def _coerce_page_size(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(_MAX_MARGINALIA_RESULTS_PER_PAGE, size))


__all__ = ["MarginaliaProvider", "MarginaliaProviderConfig"]
