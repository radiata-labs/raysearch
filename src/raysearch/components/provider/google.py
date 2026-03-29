"""Google web search provider using the public search result HTML.

This provider follows the common Google-engine pattern used in search stacks:

- it builds Google search URLs with language, country, and safesearch controls
- it applies Google ``tbs`` date filters when date bounds are available
- it detects consent, sorry, and unusual-traffic pages
- it parses public search-result blocks and unwraps Google redirect links

Configuration
=============

Example configuration in this project:

.. code:: yaml

   google:
     enabled: true
     cookies:
       CONSENT: "YES+"

Notes
=====

- Google HTML structure is unstable, so result parsing is intentionally
  defensive and selector-based.
- Date filtering is translated into Google ``qdr`` or custom-date ``tbs``
  arguments depending on the request.
- Requests may still be blocked by Google despite consent cookies and a mobile
  user agent.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import override
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import Field, field_validator

from raysearch.components.http.base import HttpClientBase
from raysearch.components.provider.base import (
    ProviderConfigBase,
    ProviderMeta,
    SearchProviderBase,
)
from raysearch.dependencies import Depends
from raysearch.models.components.provider import SearchProviderResult
from raysearch.utils import clean_whitespace

_GOOGLE_BLOCK_MARKERS = (
    "unusual traffic",
    "sorry.google.com",
    "/sorry/",
    "detected unusual traffic",
    "our systems have detected",
)
_GOOGLE_RESULT_SELECTORS = ("div.MjjYud", "div.g")
_GOOGLE_SNIPPET_SELECTORS = ('[data-sncf="1"]',)
_DEFAULT_GOOGLE_BASE_URL = "https://www.google.com/search"
_DEFAULT_GOOGLE_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 11; Pixel 5 Build/RQ3A.210905.001; wv) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 "
    "Chrome/120.0.0.0 Mobile Safari/537.36 GSA/14.48.26.29.arm64"
)


class GoogleProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "google"

    base_url: str = _DEFAULT_GOOGLE_BASE_URL
    user_agent: str = _DEFAULT_GOOGLE_USER_AGENT
    country: str = "US"
    cookies: dict[str, str] = Field(default_factory=lambda: {"CONSENT": "YES+"})

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
        if env.get("GOOGLE_BASE_URL"):
            payload["base_url"] = env["GOOGLE_BASE_URL"]
        return payload


class GoogleProvider(
    SearchProviderBase[GoogleProviderConfig],
    meta=ProviderMeta(
        name="google",
        website="https://www.google.com/",
        description="General web search with broad public web coverage and strong relevance ranking.",
        preference="Prefer broad natural-language web queries, current topics, products, brands, and open-ended exploratory searches.",
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

        page_size = self._coerce_page_size(limit)
        start_date, end_date = self._coarse_published_date_bounds(
            start_published_date=start_published_date,
            end_published_date=end_published_date,
        )
        request = self._build_request(
            query=normalized_query,
            limit=page_size,
            language=language,
            location=location,
            start=self._coerce_start(kwargs.get("start")),
            safe=self._resolve_safe(
                value=kwargs.get("safe"),
                moderation=moderation,
            ),
            country=self._resolve_country(kwargs.get("country")),
            time_range=self._resolve_time_range(kwargs.get("time_range")),
            start_date=start_date,
            end_date=end_date,
        )
        resp = await self.http.client.get(
            request["url"],
            headers=request["headers"],
            cookies=request["cookies"],
            timeout=httpx.Timeout(self.config.timeout_s),
            follow_redirects=bool(self.config.allow_redirects),
        )
        resp.raise_for_status()
        self._raise_if_blocked(resp)
        return self._trim_results(
            self._parse_search_results(resp.text, base_url=str(resp.url)),
            limit=page_size,
        )

    def _build_request(
        self,
        *,
        query: str,
        limit: int | None,
        language: str,
        location: str,
        start: int,
        safe: str,
        country: str,
        time_range: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        google_info = self._get_google_info(
            language=language,
            default_country=(location or country),
        )
        page_size = max(1, min(100, int(limit))) if limit is not None else 10
        query_params: dict[str, str | int] = {
            "q": query,
            **google_info["params"],
            "filter": "0",
            "start": max(0, int(start)),
            "num": page_size,
        }
        tbs_value = self._resolve_google_tbs(
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
        )
        if tbs_value:
            query_params["source"] = "lnt"
            query_params["tbs"] = tbs_value
        elif time_range:
            query_params["tbs"] = f"qdr:{time_range}"
        query_url = str(google_info["base_url"]) + "?" + urlencode(query_params)
        if safe in {"medium", "high"}:
            query_url += "&" + urlencode({"safe": safe})
        return {
            "url": query_url,
            "headers": google_info["headers"],
            "cookies": google_info["cookies"],
        }

    def _get_google_info(
        self,
        *,
        language: str,
        default_country: str,
    ) -> dict[str, Any]:
        cfg = self.config
        requested_language = language or "en"
        parts = [part for part in requested_language.split("-") if part]
        lang = clean_whitespace(parts[0]).lower() or "en"
        country = default_country or "US"
        lr = f"lang_{lang}"
        if lang == "zh" and country:
            lr = f"lang_{lang}-{country}"
        hl = requested_language if len(parts) > 1 else f"{lang}-{country}"
        headers = dict(cfg.headers or {})
        headers["Accept"] = "*/*"
        headers["User-Agent"] = self._google_user_agent()
        cookies = dict(cfg.cookies or {})
        cookies["CONSENT"] = "YES+"
        return {
            "base_url": clean_whitespace(str(cfg.base_url or _DEFAULT_GOOGLE_BASE_URL)),
            "headers": headers,
            "cookies": cookies,
            "params": {
                "hl": hl,
                "lr": lr,
                "gl": country,
                "cr": f"country{country}",
                "ie": "utf8",
                "oe": "utf8",
            },
        }

    def _resolve_google_tbs(
        self,
        *,
        time_range: str,
        start_date: str,
        end_date: str,
    ) -> str:
        if start_date or end_date:
            return self._build_custom_date_tbs(
                start_date=start_date,
                end_date=end_date,
            )
        if time_range:
            return f"qdr:{time_range}"
        return ""

    def _build_custom_date_tbs(self, *, start_date: str, end_date: str) -> str:
        return ",".join(
            [
                "cdr:1",
                f"cd_min:{self._format_google_date(start_date) or 'x'}",
                f"cd_max:{self._format_google_date(end_date) or 'x'}",
            ]
        )

    def _format_google_date(self, value: str) -> str:
        token = clean_whitespace(value)
        if len(token) != 10 or token.count("-") != 2:
            return ""
        year, month, day = token.split("-", 2)
        if not year or not month or not day:
            return ""
        return f"{int(month)}/{int(day)}/{year}"

    def _parse_search_results(
        self,
        raw_html: str,
        *,
        base_url: str,
    ) -> list[SearchProviderResult]:
        soup = BeautifulSoup(raw_html, "html.parser")
        results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()
        for block in self._iter_result_blocks(soup):
            title = self._extract_title(block)
            if not title:
                continue
            anchor = block.select_one("a[href]")
            if anchor is None:
                continue
            resolved_url = self._resolve_result_url(
                str(anchor.get("href", "")),
                base_url=base_url,
            )
            if not resolved_url:
                continue
            key = resolved_url.casefold()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            results.append(
                SearchProviderResult(
                    url=resolved_url,
                    title=title,
                    snippet=self._extract_snippet(block),
                    engine=self.config.name,
                )
            )
        return results

    def _iter_result_blocks(self, soup: BeautifulSoup) -> list[Tag]:
        out: list[Tag] = []
        seen: set[int] = set()
        for selector in _GOOGLE_RESULT_SELECTORS:
            for block in soup.select(selector):
                if not isinstance(block, Tag):
                    continue
                marker = id(block)
                if marker in seen:
                    continue
                seen.add(marker)
                out.append(block)
        return out

    def _extract_title(self, block: Tag) -> str:
        for selector in ('div[role*="link"]', "h3", '[role="heading"]'):
            node = block.select_one(selector)
            if node is None:
                continue
            title = clean_whitespace(node.get_text(" ", strip=True))
            if title:
                return title
        return ""

    def _extract_snippet(self, block: Tag) -> str:
        parts: list[str] = []
        for selector in _GOOGLE_SNIPPET_SELECTORS:
            for node in block.select(selector):
                if not isinstance(node, Tag):
                    continue
                for script in node.select("script"):
                    script.decompose()
                snippet = clean_whitespace(node.get_text(" ", strip=True))
                if snippet:
                    parts.append(snippet)
            if parts:
                break
        return clean_whitespace(" ".join(parts))

    def _resolve_result_url(self, raw_href: str, *, base_url: str) -> str:
        href = clean_whitespace(raw_href)
        if not href:
            return ""
        if href.startswith("/url?q="):
            href = unquote(href[7:].split("&sa=U")[0])
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if self._is_google_host(parsed.netloc) and parsed.path == "/url":
            target = parse_qs(parsed.query).get("q", [""])[0]
            href = clean_whitespace(target)
            if not href:
                return ""
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or self._is_google_host(
            parsed.netloc
        ):
            return ""
        return absolute

    def _raise_if_blocked(self, resp: httpx.Response) -> None:
        parsed = urlparse(str(resp.url))
        text = resp.text.casefold()
        if parsed.netloc == "sorry.google.com" or parsed.path.startswith("/sorry"):
            raise RuntimeError("google blocked the request with a captcha page")
        if any(marker in text for marker in _GOOGLE_BLOCK_MARKERS):
            raise RuntimeError("google blocked the request or requires verification")

    def _google_user_agent(self) -> str:
        configured = clean_whitespace(str(self.config.user_agent or ""))
        return configured or _DEFAULT_GOOGLE_USER_AGENT

    def _resolve_safe(self, value: Any, *, moderation: bool) -> str:
        token = clean_whitespace(str(value or "")).casefold()
        if token in {"medium", "high", "off"}:
            return token
        if not moderation:
            return "off"
        return "medium"

    def _resolve_country(self, value: Any) -> str:
        token = clean_whitespace(str(value or self.config.country or "")).upper()
        if len(token) == 2:
            return token
        return "US"

    def _resolve_time_range(self, value: Any) -> str:
        token = clean_whitespace(str(value or "")).casefold()
        return {
            "day": "d",
            "week": "w",
            "month": "m",
            "year": "y",
        }.get(token, "")

    def _coerce_start(self, value: Any) -> int:
        try:
            start = int(value)
        except Exception:
            return 0
        return max(0, start)

    def _coerce_page_size(self, value: int | None) -> int:
        if value is None:
            return 10
        return max(1, min(100, int(value)))

    def _trim_results(
        self,
        values: list[SearchProviderResult],
        *,
        limit: int | None,
    ) -> list[SearchProviderResult]:
        if limit is None:
            return values
        return values[: max(1, int(limit))]

    def _is_google_host(self, host: str) -> bool:
        normalized = clean_whitespace(host).lower()
        return (
            normalized == "google.com"
            or normalized.endswith(".google.com")
            or ".google." in normalized
        )


__all__ = ["GoogleProvider", "GoogleProviderConfig"]
