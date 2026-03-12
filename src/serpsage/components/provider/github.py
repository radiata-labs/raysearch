from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from typing_extensions import override

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


class GitHubProvider(SearchProviderBase[GitHubProviderConfig]):
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
        sort = clean_whitespace(str(extra.pop("sort", cfg.sort) or "")) or cfg.sort
        order = clean_whitespace(str(extra.pop("order", cfg.order) or "")) or cfg.order
        per_page = self._coerce_per_page(extra.pop("per_page", cfg.results_per_page))
        headers = dict(cfg.headers or {})
        headers["Accept"] = str(cfg.accept_header or _DEFAULT_GITHUB_ACCEPT)
        headers["User-Agent"] = str(cfg.user_agent or _DEFAULT_GITHUB_USER_AGENT)
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
            headers.setdefault("X-GitHub-Api-Version", "2022-11-28")

        payload: dict[str, Any] = {}
        final_url = str(cfg.base_url)
        used_query = normalized_query
        for candidate_query in self._candidate_queries(normalized_query):
            params = self._build_params(
                query=candidate_query,
                page=page,
                per_page=per_page,
                sort=sort,
                order=order,
                extra=extra,
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
            final_url = str(resp.url)
            used_query = candidate_query
            items = payload.get("items", [])
            if isinstance(items, list) and items:
                break
        items = payload.get("items", [])
        results: list[SearchProviderResult] = []
        if isinstance(items, list):
            for index, raw in enumerate(items, start=1):
                if not isinstance(raw, dict):
                    continue
                url = clean_whitespace(str(raw.get("html_url") or ""))
                if not url:
                    continue
                title = clean_whitespace(
                    str(raw.get("full_name") or raw.get("name") or "")
                )
                description = clean_whitespace(str(raw.get("description") or ""))
                repo_language = clean_whitespace(str(raw.get("language") or ""))
                snippet = " / ".join(
                    part for part in (repo_language, description) if part
                )
                license_info = raw.get("license") or {}
                license_name = ""
                license_url = ""
                if isinstance(license_info, dict):
                    license_name = clean_whitespace(str(license_info.get("name") or ""))
                    spdx_id = clean_whitespace(str(license_info.get("spdx_id") or ""))
                    if spdx_id and spdx_id != "NOASSERTION":
                        license_url = f"https://spdx.org/licenses/{spdx_id}.html"
                metadata = {
                    "thumbnail": clean_whitespace(
                        str((raw.get("owner") or {}).get("avatar_url") or "")
                    ),
                    "package_name": clean_whitespace(str(raw.get("name") or "")),
                    "maintainer": clean_whitespace(
                        str((raw.get("owner") or {}).get("login") or "")
                    ),
                    "published_date": self._parse_github_datetime(
                        raw.get("updated_at") or raw.get("created_at")
                    ),
                    "tags": (
                        [clean_whitespace(str(tag)) for tag in raw.get("topics", [])]
                        if isinstance(raw.get("topics"), list)
                        else []
                    ),
                    "popularity": raw.get("stargazers_count"),
                    "license_name": license_name,
                    "license_url": license_url,
                    "homepage": clean_whitespace(str(raw.get("homepage") or "")),
                    "source_code_url": clean_whitespace(
                        str(raw.get("clone_url") or "")
                    ),
                    "default_branch": clean_whitespace(
                        str(raw.get("default_branch") or "")
                    ),
                    "forks_count": raw.get("forks_count"),
                    "open_issues_count": raw.get("open_issues_count"),
                    "watchers_count": raw.get("watchers_count"),
                }
                results.append(
                    SearchProviderResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                        display_url=clean_whitespace(str(raw.get("full_name") or "")),
                        source_engine="github",
                        position=index,
                        metadata={
                            key: value
                            for key, value in metadata.items()
                            if self._keep_metadata_value(value)
                        },
                    )
                )

        total_count = payload.get("total_count")
        return SearchProviderResponse(
            provider_backend="github",
            query=normalized_query,
            page=int(page),
            language=normalized_language,
            total_results=(
                int(total_count) if isinstance(total_count, (int, float)) else None
            ),
            results=results,
            metadata={
                "request_url": final_url,
                "github_domain": str(cfg.base_url),
                "incomplete_results": bool(payload.get("incomplete_results")),
                "effective_query": used_query,
            },
        )

    def _build_params(
        self,
        *,
        query: str,
        page: int,
        per_page: int,
        sort: str,
        order: str,
        extra: dict[str, Any],
    ) -> dict[str, str]:
        params: dict[str, str] = {
            "q": query,
            "sort": sort,
            "order": order,
            "page": str(max(1, int(page))),
            "per_page": str(per_page),
        }
        for key, value in extra.items():
            token = clean_whitespace(str(key or ""))
            if not token or value is None:
                continue
            params[token] = str(value)
        return params

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

    def _coerce_per_page(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(100, size))

    def _keep_metadata_value(self, value: Any) -> bool:
        return value is not None and value not in ("", [])

    def _parse_github_datetime(self, value: Any) -> str:
        token = clean_whitespace(str(value or ""))
        if not token:
            return ""
        try:
            parsed = datetime.fromisoformat(token)
        except ValueError:
            return token
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).isoformat()


__all__ = ["GitHubProvider", "GitHubProviderConfig"]
