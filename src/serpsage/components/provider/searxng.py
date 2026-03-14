"""SearXNG meta-search provider for querying an upstream SearXNG instance.

This provider is a thin adapter around the upstream SearXNG JSON API:

- it sends queries to a configured SearXNG instance with ``format=json``
- it forwards selected extra parameters such as engines, categories, and time
  range
- it normalizes heterogeneous upstream results into a shared result schema
- it can optionally authenticate with a bearer token

Configuration
=============

Example configuration in this project:

.. code:: yaml

   searxng:
     enabled: true
     base_url: https://searx.be/search
     allow_redirects: false

Notes
=====

- This provider is useful when you want broad engine coverage without adding a
  dedicated adapter for each upstream search engine.
- SearXNG may return mixed result types and inconsistent date fields, so the
  provider normalizes several possible published-date keys.
- Extra request parameters are intentionally whitelisted to keep behavior
  predictable.
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

_DEFAULT_SEARXNG_BASE_URL = "https://searx.be/search"


class SearxngProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "searxng"

    base_url: str = _DEFAULT_SEARXNG_BASE_URL
    api_key: str | None = None
    user_agent: str = ""

    @field_validator("api_key")
    @classmethod
    def _normalize_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        token = clean_whitespace(str(value))
        return token or None

    @field_validator("user_agent")
    @classmethod
    def _normalize_user_agent(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @classmethod
    @override
    def inject_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> dict[str, Any]:
        payload = dict(raw)
        if env.get("SEARXNG_BASE_URL"):
            payload["base_url"] = env["SEARXNG_BASE_URL"]
        api_key = env.get("SEARXNG_API_KEY")
        if api_key:
            payload["api_key"] = api_key
        cf_id = env.get("SEARXNG_CF_ACCESS_CLIENT_ID")
        cf_secret = env.get("SEARXNG_CF_ACCESS_CLIENT_SECRET")
        if cf_id and cf_secret:
            payload.setdefault("headers", {})
            payload["headers"].setdefault("CF-Access-Client-Id", cf_id)
            payload["headers"].setdefault("CF-Access-Client-Secret", cf_secret)
        return payload


class SearxngProvider(
    SearchProviderBase[SearxngProviderConfig],
    meta=ProviderMeta(
        name="searxng",
        website="https://searxng.org/",
        description="Meta-search through a SearXNG instance that can aggregate multiple external search engines.",
        preference="Prefer broad web queries when a meta-search route is useful for mixed-source discovery across multiple engines.",
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
        cfg = self.config
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        payload: dict[str, str] = {"q": normalized_query, "format": "json"}
        page_size = max(1, int(limit)) if limit is not None else None
        if page_size is not None:
            payload["limit"] = str(page_size)
        if language:
            payload["language"] = str(language)
        effective_kwargs = dict(kwargs)
        effective_kwargs.setdefault(
            "safesearch",
            self._resolve_safesearch_value(
                value=effective_kwargs.get("safesearch"),
                moderation=moderation,
            ),
        )
        if not clean_whitespace(str(effective_kwargs.get("time_range") or "")):
            effective_kwargs["time_range"] = self._relative_time_range_from_bounds(
                start_published_date=start_published_date,
                end_published_date=end_published_date,
            )
        self._merge_extra_payload(payload=payload, extra=effective_kwargs)
        resp = await self.http.client.get(
            cfg.base_url,
            params=payload,
            headers=self._build_headers(),
            timeout=httpx.Timeout(cfg.timeout_s),
            follow_redirects=bool(cfg.allow_redirects),
        )
        resp.raise_for_status()
        data = resp.json()
        results = self._parse_results(data.get("results"))
        results = self._filter_results_by_published_date(
            results=results,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
            include_undated=True,
        )
        return results[:page_size] if page_size is not None else results

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        if self.config.user_agent:
            headers.setdefault("User-Agent", str(self.config.user_agent))
        if self.config.api_key:
            headers.setdefault("Authorization", f"Bearer {self.config.api_key}")
        return headers

    def _parse_results(self, raw_results: Any) -> list[SearchProviderResult]:
        results: list[SearchProviderResult] = []
        for raw in raw_results if isinstance(raw_results, list) else []:
            if not isinstance(raw, dict):
                continue
            url = clean_whitespace(str(raw.get("url") or ""))
            if not url:
                continue
            snippet = raw.get("content")
            if snippet is None:
                snippet = raw.get("description")
            results.append(
                SearchProviderResult(
                    url=url,
                    title=clean_whitespace(str(raw.get("title") or "")),
                    snippet=clean_whitespace(str(snippet or "")),
                    engine=self.config.name,
                    published_date=self._parse_published_date(raw),
                )
            )
        return results

    def _parse_published_date(self, raw: dict[str, Any]) -> str:
        for key in (
            "publishedDate",
            "published_date",
            "published",
            "pubdate",
            "created_at",
            "date",
        ):
            token = clean_whitespace(str(raw.get(key) or ""))
            if not token:
                continue
            try:
                return normalize_iso8601_string(token)
            except ValueError:
                continue
        return ""

    def _merge_extra_payload(
        self,
        *,
        payload: dict[str, str],
        extra: dict[str, Any],
    ) -> None:
        allowed_keys = {
            "categories",
            "engines",
            "pageno",
            "safesearch",
            "time_range",
        }
        for key in allowed_keys:
            raw_value = extra.get(key)
            if raw_value is None:
                continue
            if isinstance(raw_value, (list, tuple, set)):
                token = ",".join(
                    clean_whitespace(str(item or ""))
                    for item in raw_value
                    if clean_whitespace(str(item or ""))
                )
            else:
                token = clean_whitespace(str(raw_value or ""))
            if token:
                payload[key] = token

    def _resolve_safesearch_value(self, value: Any, *, moderation: bool) -> str:
        token = clean_whitespace(str(value or ""))
        if token in {"0", "1", "2"}:
            return token
        if isinstance(value, bool):
            return "1" if value else "0"
        return "1" if moderation else "0"


__all__ = ["SearxngProvider", "SearxngProviderConfig"]
