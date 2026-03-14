"""GitHub repository search provider backed by the official REST API.

This provider mirrors the behavior expected from a repository-search adapter:

- it queries the GitHub search repositories endpoint
- it supports optional authenticated requests for better rate limits
- it preserves GitHub sort and order controls
- it can narrow by repository creation date when date bounds are provided

Configuration
=============

Example configuration in this project:

.. code:: yaml

   github:
     enabled: true
     base_url: https://api.github.com/search/repositories
     sort: stars
     order: desc
     results_per_page: 10

Notes
=====

- Auth is optional, but providing a token generally improves reliability.
- The provider emits repository creation time as ``published_date`` because that
  is the only stable publication-like timestamp exposed by the search response.
- Query fallback logic tries progressively narrower tokens when a long natural
  language query returns no repository hits.
"""

from __future__ import annotations

import re
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

_DEFAULT_GITHUB_BASE_URL = "https://api.github.com/search/repositories"
_DEFAULT_GITHUB_ACCEPT = "application/vnd.github.preview.text-match+json"
_DEFAULT_GITHUB_USER_AGENT = "serpsage-github-provider/1.0"
_DEFAULT_GITHUB_SORT = "stars"
_DEFAULT_GITHUB_ORDER = "desc"
_GITHUB_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class GitHubProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "github"

    base_url: str = _DEFAULT_GITHUB_BASE_URL
    api_key: str | None = None
    user_agent: str = _DEFAULT_GITHUB_USER_AGENT
    accept_header: str = _DEFAULT_GITHUB_ACCEPT
    sort: str = _DEFAULT_GITHUB_SORT
    order: str = _DEFAULT_GITHUB_ORDER
    results_per_page: int = 10

    @field_validator("api_key")
    @classmethod
    def _normalize_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        token = clean_whitespace(str(value))
        return token or None

    @field_validator("user_agent", "accept_header", "sort", "order")
    @classmethod
    def _normalize_text_fields(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("github results_per_page must be > 0")
        if size > 100:
            raise ValueError("github results_per_page must be <= 100")
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
        if env.get("GITHUB_BASE_URL"):
            payload["base_url"] = env["GITHUB_BASE_URL"]
        api_key = env.get("GITHUB_API_KEY") or env.get("GITHUB_TOKEN")
        if api_key:
            payload["api_key"] = api_key
        return payload


class GitHubProvider(
    SearchProviderBase[GitHubProviderConfig],
    meta=ProviderMeta(
        name="github",
        website="https://github.com/",
        description="Repository search across GitHub projects, maintainers, languages, and open source metadata.",
        preference="Prefer repository names, library names, package identifiers, programming tools, frameworks, and code-focused technical queries.",
        categories=["it", "repos"],
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

        headers = self._build_headers()
        per_page = self._coerce_per_page(
            limit if limit is not None else cfg.results_per_page
        )
        page = self._coerce_page(kwargs.get("page"))
        sort = clean_whitespace(str(kwargs.get("sort") or cfg.sort or "")) or cfg.sort
        order = (
            clean_whitespace(str(kwargs.get("order") or cfg.order or "")) or cfg.order
        )
        start_date, end_date = self._coarse_published_date_bounds(
            start_published_date=start_published_date,
            end_published_date=end_published_date,
        )

        results: list[SearchProviderResult] = []
        for candidate_query in self._candidate_queries(normalized_query):
            params = self._build_params(
                query=self._apply_created_date_qualifiers(
                    query=candidate_query,
                    start_date=start_date,
                    end_date=end_date,
                ),
                page=page,
                per_page=per_page,
                sort=sort,
                order=order,
            )
            resp = await self.http.client.get(
                str(cfg.base_url),
                params=params,
                headers=headers,
                timeout=httpx.Timeout(cfg.timeout_s),
                follow_redirects=bool(cfg.allow_redirects),
            )
            resp.raise_for_status()
            payload = resp.json()
            results = self._parse_results(payload.get("items"))
            if results:
                break
        results = self._filter_results_by_published_date(
            results=results,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
        )
        return results[:per_page]

    def _build_headers(self) -> dict[str, str]:
        cfg = self.config
        headers = dict(cfg.headers or {})
        headers["Accept"] = str(cfg.accept_header or _DEFAULT_GITHUB_ACCEPT)
        headers["User-Agent"] = str(cfg.user_agent or _DEFAULT_GITHUB_USER_AGENT)
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
            headers.setdefault("X-GitHub-Api-Version", "2022-11-28")
        return headers

    def _parse_results(self, items: Any) -> list[SearchProviderResult]:
        results: list[SearchProviderResult] = []
        for raw in items if isinstance(items, list) else []:
            if not isinstance(raw, dict):
                continue
            url = clean_whitespace(str(raw.get("html_url") or ""))
            if not url:
                continue
            title = clean_whitespace(str(raw.get("full_name") or raw.get("name") or ""))
            description = clean_whitespace(str(raw.get("description") or ""))
            language = clean_whitespace(str(raw.get("language") or ""))
            results.append(
                SearchProviderResult(
                    url=url,
                    title=title,
                    snippet=" / ".join(
                        part for part in (language, description) if part
                    ),
                    engine=self.config.name,
                    published_date=self._parse_published_date(raw.get("created_at")),
                )
            )
        return results

    def _build_params(
        self,
        *,
        query: str,
        page: int,
        per_page: int,
        sort: str,
        order: str,
    ) -> dict[str, str]:
        return {
            "q": query,
            "sort": sort,
            "order": order,
            "page": str(max(1, int(page))),
            "per_page": str(per_page),
        }

    def _apply_created_date_qualifiers(
        self,
        *,
        query: str,
        start_date: str,
        end_date: str,
    ) -> str:
        parts = [clean_whitespace(query)]
        if start_date:
            parts.append(f"created:>={start_date}")
        if end_date:
            parts.append(f"created:<{end_date}")
        return " ".join(part for part in parts if part)

    def _candidate_queries(self, query: str) -> list[str]:
        primary = clean_whitespace(query)
        if not primary:
            return []
        out: list[str] = [primary]
        seen = {primary.casefold()}
        tokens = [
            clean_whitespace(match.group(0))
            for match in _GITHUB_QUERY_TOKEN_RE.finditer(primary)
        ]
        preferred = [
            token
            for token in tokens
            if len(token) >= 3 and any(ch.isdigit() or ch in "._-" for ch in token)
        ]
        fallback = [token for token in tokens if len(token) >= 3]
        for candidate in preferred + list(reversed(fallback)):
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        return out

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
        return max(1, min(100, size))

    def _coerce_page(self, value: Any) -> int:
        try:
            page = int(value)
        except Exception:
            return 1
        return max(1, page)


__all__ = ["GitHubProvider", "GitHubProviderConfig"]
