from __future__ import annotations

import re
from typing import Any
from typing_extensions import override
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag
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

_DDG_BLOCK_MARKERS = (
    "challenge-form",
    "anomaly.js",
    "prove you are human",
    "not a robot",
)
_DDG_RESULT_SELECTORS = (
    "div#links > div.web-result",
    "div#links > div.result.results_links",
)
_DDG_SNIPPET_SELECTORS = (
    "a.result__snippet",
    "div.result__snippet",
)
_DEFAULT_DDG_BASE_URL = "https://html.duckduckgo.com/html/"
_DEFAULT_DDG_REFERER = _DEFAULT_DDG_BASE_URL
_DEFAULT_DDG_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_DDG_JINA_LITE_URL = "https://r.jina.ai/http://lite.duckduckgo.com/lite/"
_DDG_JINA_RESULT_RE = re.compile(
    r"^\d+\.\[(?P<title>.+?)\]\((?P<url>.+?)\)\s*(?P<snippet>.*)$"
)


class DuckDuckGoProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "duckduckgo"

    base_url: str = _DEFAULT_DDG_BASE_URL
    user_agent: str = _DEFAULT_DDG_USER_AGENT
    region: str = "wt-wt"

    @field_validator("user_agent", "region")
    @classmethod
    def _normalize_text_fields(cls, value: str) -> str:
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
        if env.get("DUCKDUCKGO_BASE_URL"):
            payload["base_url"] = env["DUCKDUCKGO_BASE_URL"]
        return payload


class DuckDuckGoProvider(
    SearchProviderBase[DuckDuckGoProviderConfig],
    meta=ProviderMeta(
        name="duckduckgo",
        website="https://duckduckgo.com/",
        description="General web search from DuckDuckGo with privacy-oriented results and lightweight HTML endpoints.",
        preference="Prefer concise web queries, lightweight factual lookups, and general web searches that do not need provider-specific verticals.",
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
        locale: str = "",
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        cfg = self.config
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")
        if len(normalized_query) >= 500:
            return []

        request_base_url = self._request_base_url()
        request_params = self._build_request_params(
            query=normalized_query,
            region=kwargs.get("region"),
            time_range=self._resolve_time_range(
                runtime_value=kwargs.get("time_range"),
                start_published_date=start_published_date,
                end_published_date=end_published_date,
            ),
        )
        try:
            resp = await self.http.client.get(
                request_base_url,
                params=request_params,
                headers=self._build_headers(),
                timeout=httpx.Timeout(cfg.timeout_s),
                follow_redirects=bool(cfg.allow_redirects),
            )
            resp.raise_for_status()
            self._raise_if_blocked(resp)
            results = self._parse_search_results(resp.text, base_url=request_base_url)
            if not results:
                results = await self._search_via_jina_mirror(query=normalized_query)
        except Exception:
            results = await self._search_via_jina_mirror(query=normalized_query)
        return self._trim_results(results, limit=limit)

    def _resolve_time_range(
        self,
        *,
        runtime_value: Any,
        start_published_date: str | None,
        end_published_date: str | None,
    ) -> str:
        token = clean_whitespace(str(runtime_value or "")).casefold()
        if token:
            return token
        return self._relative_time_range_from_bounds(
            start_published_date=start_published_date,
            end_published_date=end_published_date,
        )

    def _build_request_params(
        self,
        *,
        query: str,
        region: Any,
        time_range: str,
    ) -> dict[str, str]:
        params: dict[str, str] = {"q": self._quote_bangs(query)}
        normalized_region = clean_whitespace(
            str(region or self.config.region or "")
        ).lower()
        if normalized_region and normalized_region != "all":
            params["kl"] = normalized_region
        time_map = {"day": "d", "week": "w", "month": "m", "year": "y"}
        if time_range in time_map:
            params["df"] = time_map[time_range]
        return params

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers["User-Agent"] = str(self.config.user_agent or _DEFAULT_DDG_USER_AGENT)
        headers.setdefault(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        headers["Referer"] = _DEFAULT_DDG_REFERER
        return headers

    def _request_base_url(self) -> str:
        base_url = clean_whitespace(str(self.config.base_url or _DEFAULT_DDG_BASE_URL))
        if not base_url:
            return _DEFAULT_DDG_BASE_URL
        if base_url.endswith("/html"):
            return base_url + "/"
        return base_url

    async def _search_via_jina_mirror(
        self,
        *,
        query: str,
    ) -> list[SearchProviderResult]:
        resp = await self.http.client.get(
            _DDG_JINA_LITE_URL + "?q=" + quote_plus(query),
            timeout=httpx.Timeout(self.config.timeout_s),
            follow_redirects=bool(self.config.allow_redirects),
        )
        resp.raise_for_status()
        return self._parse_jina_lite_results(resp.text)

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
            for script in block.select("script, style"):
                script.decompose()
            anchor = block.select_one("h2 a[href], a.result__a[href], a[href]")
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
            title = self._extract_title(block)
            snippet = self._extract_snippet(block)
            if not title and not snippet:
                continue
            seen_urls.add(key)
            results.append(
                SearchProviderResult(
                    url=resolved_url,
                    title=title,
                    snippet=snippet,
                    engine=self.config.name,
                )
            )
        return results

    def _iter_result_blocks(self, soup: BeautifulSoup) -> list[Tag]:
        out: list[Tag] = []
        seen: set[int] = set()
        for selector in _DDG_RESULT_SELECTORS:
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
        for selector in ("h2 a", "a.result__a", "h2"):
            node = block.select_one(selector)
            if node is None:
                continue
            title = clean_whitespace(node.get_text(" ", strip=True))
            if title:
                return title
        return ""

    def _extract_snippet(self, block: Tag) -> str:
        for selector in _DDG_SNIPPET_SELECTORS:
            node = block.select_one(selector)
            if node is None:
                continue
            snippet = clean_whitespace(node.get_text(" ", strip=True))
            if snippet:
                return snippet
        return ""

    def _resolve_result_url(self, raw_href: str, *, base_url: str) -> str:
        href = clean_whitespace(raw_href)
        if not href:
            return ""
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if self._is_ddg_host(parsed.netloc) and parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if not target:
                target = parse_qs(parsed.query).get("u", [""])[0]
            href = clean_whitespace(unquote(target))
            if not href:
                return ""
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or self._is_ddg_host(parsed.netloc):
            return ""
        return absolute

    def _parse_jina_lite_results(self, text: str) -> list[SearchProviderResult]:
        results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()
        for raw_line in str(text or "").splitlines():
            line = clean_whitespace(raw_line)
            if not line:
                continue
            match = _DDG_JINA_RESULT_RE.match(line)
            if match is None:
                continue
            resolved_url = self._resolve_result_url(
                match.group("url"),
                base_url=_DEFAULT_DDG_BASE_URL,
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
                    title=clean_whitespace(match.group("title")),
                    snippet=clean_whitespace(match.group("snippet")),
                    engine=self.config.name,
                )
            )
        return results

    def _raise_if_blocked(self, resp: httpx.Response) -> None:
        parsed = urlparse(str(resp.url))
        text = resp.text.casefold()
        if parsed.path.startswith("/anomaly") or 'id="challenge-form"' in text:
            raise RuntimeError("duckduckgo blocked the request with a challenge page")
        if any(marker in text for marker in _DDG_BLOCK_MARKERS):
            raise RuntimeError(
                "duckduckgo blocked the request or requires verification"
            )

    def _quote_bangs(self, query: str) -> str:
        parts: list[str] = []
        for value in re.split(r"(\s+)", query):
            token = clean_whitespace(value)
            if not token:
                continue
            if token.startswith("!") and len(token) > 1:
                token = f"'{token}'"
            parts.append(token)
        return " ".join(parts)

    def _is_ddg_host(self, host: str) -> bool:
        normalized = clean_whitespace(host).lower()
        return normalized == "duckduckgo.com" or normalized.endswith(".duckduckgo.com")

    def _trim_results(
        self,
        values: list[SearchProviderResult],
        *,
        limit: int | None,
    ) -> list[SearchProviderResult]:
        if limit is None:
            return values
        return values[: max(1, int(limit))]


__all__ = ["DuckDuckGoProvider", "DuckDuckGoProviderConfig"]
