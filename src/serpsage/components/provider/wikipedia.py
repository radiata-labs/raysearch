"""Wikipedia search provider based on the Wikimedia REST search APIs.

This provider follows the expected shape of an encyclopedia search adapter:

- it searches article titles through the Wikimedia REST search endpoint
- it optionally expands results with page summaries from the summary endpoint
- it supports per-language Wikipedia domains with English as a fallback default
- it extracts description and timestamp fields into normalized search results

Configuration
=============

Example configuration in this project:

.. code:: yaml

   wikipedia:
     enabled: true
     base_url: https://{wiki_netloc}/w/rest.php/v1/search/title
     summary_url_template: https://{wiki_netloc}/api/rest_v1/page/summary/{title}
     page_url_template: https://{wiki_netloc}/wiki/{title}
     default_language: en
     results_per_page: 5

Notes
=====

- Wikipedia is especially strong for entities, topics, events, and reference
  lookups rather than current news.
- The provider uses REST endpoints instead of scraping article HTML.
- When summary metadata includes timestamps, they are normalized into
  ``published_date`` for downstream reuse.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import override
from urllib.parse import quote

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
from serpsage.utils import clean_whitespace, normalize_iso8601_string, strip_html

_DEFAULT_WIKIPEDIA_SEARCH_URL = "https://{wiki_netloc}/w/rest.php/v1/search/title"
_DEFAULT_WIKIPEDIA_SUMMARY_URL = (
    "https://{wiki_netloc}/api/rest_v1/page/summary/{title}"
)
_DEFAULT_WIKIPEDIA_PAGE_URL = "https://{wiki_netloc}/wiki/{title}"
_DEFAULT_WIKIPEDIA_USER_AGENT = "serpsage-wikipedia-provider/1.0 (https://example.com)"
_WIKI_LC_LOCALE_VARIANTS: dict[str, tuple[str, ...]] = {
    "zh": ("zh-cn", "zh-hk", "zh-mo", "zh-my", "zh-sg", "zh-tw"),
    "zh-classical": ("zh-classical",),
}
_WIKI_SCRIPT_VARIANTS = {
    "zh-hans": "zh",
    "zh-hant": "zh",
}
_WIKI_SPECIAL_TAGS = {
    "be-tarask",
    "nds-nl",
    "simple",
    "zh-classical",
    "zh-min-nan",
    "zh-yue",
}


class WikipediaProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "wikipedia"

    base_url: str = _DEFAULT_WIKIPEDIA_SEARCH_URL
    allow_redirects: bool = True
    user_agent: str = _DEFAULT_WIKIPEDIA_USER_AGENT
    summary_url_template: str = _DEFAULT_WIKIPEDIA_SUMMARY_URL
    page_url_template: str = _DEFAULT_WIKIPEDIA_PAGE_URL
    default_language: str = "en"
    results_per_page: int = 5

    @field_validator(
        "user_agent",
        "summary_url_template",
        "page_url_template",
        "default_language",
    )
    @classmethod
    def _normalize_text_fields(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("wikipedia results_per_page must be > 0")
        if size > 20:
            raise ValueError("wikipedia results_per_page must be <= 20")
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
        if env.get("WIKIPEDIA_BASE_URL"):
            payload["base_url"] = env["WIKIPEDIA_BASE_URL"]
        return payload


class WikipediaProvider(
    SearchProviderBase[WikipediaProviderConfig],
    meta=ProviderMeta(
        name="wikipedia",
        website="https://www.wikipedia.org/",
        description="Encyclopedia search for Wikipedia article titles, summaries, and multilingual reference pages.",
        preference="Prefer entity-centric reference queries about people, places, events, concepts, organizations, and historical topics.",
        categories=["infobox"],
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

        resolved = self._resolve_wiki_language(
            language or cfg.default_language
        )
        per_page = self._coerce_per_page(
            limit if limit is not None else cfg.results_per_page
        )
        offset = self._coerce_offset(kwargs.get("offset"))
        headers = self._build_headers(accept_language=resolved["accept_language"])
        resp = await self.http.client.get(
            self._format_template(
                str(cfg.base_url),
                wiki_netloc=resolved["wiki_netloc"],
            ),
            params=self._build_search_params(
                query=normalized_query,
                offset=offset,
                per_page=per_page,
            ),
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s),
            follow_redirects=bool(cfg.allow_redirects),
        )
        resp.raise_for_status()
        payload = resp.json()
        results = await self._build_results(
            hits=payload.get("pages") if isinstance(payload, dict) else None,
            resolved=resolved,
            headers=headers,
        )
        if (
            not results
            and int(offset) == 0
            and self._coerce_bool(kwargs.get("exact_fallback"), default=True)
        ):
            exact = await self._fetch_exact_summary(
                query=normalized_query,
                resolved=resolved,
                headers=headers,
            )
            if exact is not None:
                results = [exact]
        results = self._filter_results_by_published_date(
            results=results,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
            include_undated=True,
        )
        return results[:per_page]

    def _build_headers(self, *, accept_language: str) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers["Accept"] = "application/json"
        headers["User-Agent"] = str(
            self.config.user_agent or _DEFAULT_WIKIPEDIA_USER_AGENT
        )
        if accept_language:
            headers["Accept-Language"] = accept_language
        return headers

    def _build_search_params(
        self,
        *,
        query: str,
        offset: int,
        per_page: int,
    ) -> dict[str, str]:
        return {
            "q": query,
            "limit": str(per_page),
            "offset": str(max(0, int(offset))),
        }

    async def _build_results(
        self,
        *,
        hits: Any,
        resolved: dict[str, str],
        headers: dict[str, str],
    ) -> list[SearchProviderResult]:
        results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()
        for raw in hits if isinstance(hits, list) else []:
            if not isinstance(raw, dict):
                continue
            title = clean_whitespace(str(raw.get("title") or ""))
            if not title:
                continue
            summary = await self._fetch_summary(
                title=title,
                resolved=resolved,
                headers=headers,
            )
            item = self._build_result(
                raw=raw,
                summary=summary,
                resolved=resolved,
                fallback_title=title,
                fallback_snippet=clean_whitespace(
                    strip_html(str(raw.get("excerpt") or raw.get("description") or ""))
                ),
            )
            if item is None:
                continue
            key = item.url.casefold()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            results.append(item)
        return results

    async def _fetch_exact_summary(
        self,
        *,
        query: str,
        resolved: dict[str, str],
        headers: dict[str, str],
    ) -> SearchProviderResult | None:
        title = self._normalize_exact_title(query)
        if not title:
            return None
        summary = await self._fetch_summary(
            title=title,
            resolved=resolved,
            headers=headers,
        )
        if summary is None:
            return None
        return self._build_result(
            raw={},
            summary=summary,
            resolved=resolved,
            fallback_title=title,
            fallback_snippet="",
        )

    async def _fetch_summary(
        self,
        *,
        title: str,
        resolved: dict[str, str],
        headers: dict[str, str],
    ) -> dict[str, Any] | None:
        resp = await self.http.client.get(
            self._format_template(
                self.config.summary_url_template,
                wiki_netloc=resolved["wiki_netloc"],
                title=self._quote_summary_title(title),
            ),
            headers=headers,
            timeout=httpx.Timeout(self.config.timeout_s),
            follow_redirects=bool(self.config.allow_redirects),
        )
        if resp.status_code == 404:
            return None
        if resp.status_code == 400:
            try:
                error_payload = resp.json()
            except Exception:
                error_payload = None
            if self._is_invalid_title_error(error_payload):
                return None
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            return None
        payload = resp.json()
        return payload if isinstance(payload, dict) else None

    def _build_result(
        self,
        *,
        raw: dict[str, Any],
        summary: dict[str, Any] | None,
        resolved: dict[str, str],
        fallback_title: str,
        fallback_snippet: str,
    ) -> SearchProviderResult | None:
        url = self._extract_page_url(
            summary,
            resolved=resolved,
            fallback_title=fallback_title,
        )
        if not url:
            return None
        return SearchProviderResult(
            url=url,
            title=self._extract_display_title(
                summary,
                fallback_title=fallback_title,
            ),
            snippet=(
                clean_whitespace(str((summary or {}).get("extract") or ""))
                or clean_whitespace(str((summary or {}).get("description") or ""))
                or fallback_snippet
            ),
            engine=self.config.name,
            published_date=self._parse_published_date(raw=raw, summary=summary),
        )

    def _extract_page_url(
        self,
        summary: dict[str, Any] | None,
        *,
        resolved: dict[str, str],
        fallback_title: str,
    ) -> str:
        desktop = ((summary or {}).get("content_urls") or {}).get("desktop") or {}
        if isinstance(desktop, dict):
            page_url = clean_whitespace(str(desktop.get("page") or ""))
            if page_url:
                return page_url
        return self._format_template(
            self.config.page_url_template,
            wiki_netloc=resolved["wiki_netloc"],
            title=self._quote_page_title(fallback_title),
        )

    def _extract_display_title(
        self,
        summary: dict[str, Any] | None,
        *,
        fallback_title: str,
    ) -> str:
        titles = (summary or {}).get("titles") or {}
        if isinstance(titles, dict):
            display = clean_whitespace(str(titles.get("display") or ""))
            if display:
                return display
        title = clean_whitespace(str((summary or {}).get("title") or ""))
        return title or fallback_title

    def _parse_published_date(
        self,
        *,
        raw: dict[str, Any],
        summary: dict[str, Any] | None,
    ) -> str:
        for source in (summary or {}, raw):
            for key in (
                "timestamp",
                "rev_timestamp",
                "published_date",
                "published",
                "updated_at",
                "date",
            ):
                token = clean_whitespace(str(source.get(key) or ""))
                if not token:
                    continue
                try:
                    return normalize_iso8601_string(token)
                except ValueError:
                    continue
        return ""

    def _resolve_wiki_language(self, value: str) -> dict[str, str]:
        requested = self._normalize_language(value)
        if not requested:
            requested = (
                self._normalize_language(
                    clean_whitespace(self.config.default_language or "en")
                )
                or "en"
            )
        wiki_tag = self._wiki_tag_for_language(requested)
        return {
            "request_language": requested or wiki_tag,
            "accept_language": self._canonical_accept_language(requested or wiki_tag),
            "wiki_tag": wiki_tag,
            "wiki_netloc": f"{wiki_tag}.wikipedia.org",
        }

    def _wiki_tag_for_language(self, language: str) -> str:
        normalized = language.casefold()
        if not normalized:
            return "en"
        if normalized in _WIKI_SCRIPT_VARIANTS:
            return _WIKI_SCRIPT_VARIANTS[normalized]
        for wiki_tag, variants in _WIKI_LC_LOCALE_VARIANTS.items():
            if normalized == wiki_tag or normalized in variants:
                return wiki_tag
        if normalized in _WIKI_SPECIAL_TAGS:
            return normalized
        parts = [part for part in normalized.split("-") if part]
        return parts[0] if parts else "en"

    def _canonical_accept_language(self, value: str) -> str:
        token = clean_whitespace(value).replace("_", "-")
        parts = [part for part in token.split("-") if part]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0].lower()
        normalized_parts = [parts[0].lower()]
        for item in parts[1:]:
            if len(item) == 2:
                normalized_parts.append(item.upper())
            else:
                normalized_parts.append(item.title())
        return "-".join(normalized_parts)

    def _normalize_exact_title(self, value: str) -> str:
        title = clean_whitespace(value)
        if not title:
            return ""
        if title.islower():
            return title.title()
        return title

    def _quote_summary_title(self, title: str) -> str:
        return quote(clean_whitespace(title), safe="")

    def _quote_page_title(self, title: str) -> str:
        return quote(clean_whitespace(title).replace(" ", "_"), safe="")

    def _format_template(
        self,
        template: str,
        *,
        wiki_netloc: str,
        title: str = "",
    ) -> str:
        return str(template).format(wiki_netloc=wiki_netloc, title=title)

    def _is_invalid_title_error(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        return (
            payload.get("type")
            == "https://mediawiki.org/wiki/HyperSwitch/errors/bad_request"
            and payload.get("detail") == "title-invalid-characters"
        )

    def _coerce_per_page(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(20, size))

    def _coerce_offset(self, value: Any) -> int:
        try:
            offset = int(value)
        except Exception:
            return 0
        return max(0, offset)

    def _coerce_bool(self, value: Any, *, default: bool) -> bool:
        if value is None:
            return default
        token = clean_whitespace(str(value)).casefold()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
        return bool(value)


__all__ = ["WikipediaProvider", "WikipediaProviderConfig"]
