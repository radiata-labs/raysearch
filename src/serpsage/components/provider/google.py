from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from serpsage.components.base import ComponentMeta, Depends
from serpsage.components.http.base import HttpClientBase
from serpsage.components.provider.base import (
    GoogleSafeSearchKey,
    ProviderConfigBase,
    SearchProviderBase,
)
from serpsage.components.registry import register_component
from serpsage.models.components.provider import (
    SearchProviderResponse,
    SearchProviderResult,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from collections.abc import Iterable

_GOOGLE_TIME_RANGE_MAP = {
    "day": "qdr:d",
    "week": "qdr:w",
    "month": "qdr:m",
    "year": "qdr:y",
}
_GOOGLE_BLOCK_MARKERS = (
    "unusual traffic",
    "sorry.google.com",
    "/sorry/",
    "detected unusual traffic",
    "our systems have detected",
)
_GOOGLE_RESULT_SELECTORS = ("div.MjjYud", "div.g")
_GOOGLE_SNIPPET_SELECTORS = (
    '[data-sncf="1"]',
    "div.VwiC3b",
    "div.yXK7lf",
    "span.aCOpRe",
    "div.s3v9rd",
)
_DEFAULT_GOOGLE_BASE_URL = "https://www.google.com/search"
_DEFAULT_GOOGLE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


class GoogleProviderConfig(ProviderConfigBase):
    safe: GoogleSafeSearchKey = "off"
    country: str = "US"

    @classmethod
    @override
    def inject_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> dict[str, Any]:
        payload = dict(raw)
        payload.setdefault("base_url", _DEFAULT_GOOGLE_BASE_URL)
        payload.setdefault("user_agent", _DEFAULT_GOOGLE_USER_AGENT)
        payload.setdefault("country", "US")
        payload.setdefault("cookies", {}).setdefault("CONSENT", "YES+")
        if env.get("PROVIDER_BASE_URL"):
            payload["base_url"] = env["PROVIDER_BASE_URL"]
        elif env.get("SEARCH_BASE_URL"):
            payload["base_url"] = env["SEARCH_BASE_URL"]
        elif env.get("GOOGLE_BASE_URL"):
            payload["base_url"] = env["GOOGLE_BASE_URL"]
        api_key = env.get("PROVIDER_API_KEY") or env.get("SEARCH_API_KEY")
        if api_key:
            payload["api_key"] = api_key
        user_agent = env.get("PROVIDER_USER_AGENT") or env.get("GOOGLE_USER_AGENT")
        if user_agent:
            payload["user_agent"] = user_agent
        country = env.get("PROVIDER_COUNTRY") or env.get("GOOGLE_COUNTRY")
        if country:
            payload["country"] = country
        safe = env.get("PROVIDER_SAFE") or env.get("GOOGLE_SAFE")
        if safe:
            payload["safe"] = str(safe).strip().lower()
        results_per_page = env.get("PROVIDER_RESULTS_PER_PAGE")
        if results_per_page:
            payload["results_per_page"] = results_per_page
        return payload


_GOOGLE_META = ComponentMeta(
    family="provider",
    name="google",
    version="1.0.0",
    summary="Google HTML search provider.",
    provides=("provider.search",),
    config_model=GoogleProviderConfig,
)


@register_component(meta=_GOOGLE_META)
class GoogleProvider(SearchProviderBase[GoogleProviderConfig]):
    meta = _GOOGLE_META

    def __init__(
        self,
        *,
        rt: object,
        config: GoogleProviderConfig,
        http: HttpClientBase = Depends(),
    ) -> None:
        super().__init__(rt=rt, config=config, bound_deps=(http,))
        self._http = http.client

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
        if not normalized_query:
            raise ValueError("query must not be empty")
        request_params = self._build_request_params(
            query=normalized_query,
            page=page,
            language=language,
            kwargs=kwargs,
        )
        headers = dict(cfg.headers or {})
        headers.setdefault("Accept", "*/*")
        headers.setdefault("User-Agent", str(cfg.user_agent))
        cookies = dict(cfg.cookies or {})
        cookies.setdefault("CONSENT", "YES+")
        resp = await self._http.get(
            str(cfg.base_url),
            params=request_params,
            headers=headers,
            cookies=cookies,
            timeout=httpx.Timeout(cfg.timeout_s),
            follow_redirects=bool(cfg.allow_redirects),
        )
        resp.raise_for_status()
        self._raise_if_blocked(resp)
        results, suggestions = self._parse_search_results(
            resp.text, base_url=cfg.base_url
        )
        return SearchProviderResponse(
            provider_backend="google",
            query=normalized_query,
            page=int(page),
            language=clean_whitespace(language),
            suggestions=suggestions,
            results=results,
            metadata={
                "request_url": str(resp.url),
                "google_domain": str(cfg.base_url),
            },
        )

    def _build_request_params(
        self,
        *,
        query: str,
        page: int,
        language: str,
        kwargs: dict[str, Any],
    ) -> dict[str, str]:
        cfg = self.config
        extra = dict(kwargs)
        per_page = self._coerce_positive_int(extra.pop("num", cfg.results_per_page))
        safe = clean_whitespace(str(extra.pop("safe", cfg.safe) or "")).casefold()
        time_range = clean_whitespace(str(extra.pop("time_range", "") or "")).casefold()
        country = clean_whitespace(str(extra.pop("country", cfg.country) or "")).upper()
        ui_lang, result_lang, region = self._resolve_locale(
            language=language,
            default_country=country,
        )
        params: dict[str, str] = {
            "q": query,
            "ie": "utf8",
            "oe": "utf8",
            "filter": "0",
            "start": str(max(0, (int(page) - 1) * per_page)),
            "num": str(per_page),
            "hl": ui_lang,
        }
        if result_lang:
            params["lr"] = f"lang_{result_lang}"
        if region:
            params["gl"] = region
            params["cr"] = f"country{region}"
        if safe in {"off", "medium", "high"}:
            params["safe"] = safe
        if time_range in _GOOGLE_TIME_RANGE_MAP:
            params["tbs"] = _GOOGLE_TIME_RANGE_MAP[time_range]
        for key, value in extra.items():
            token = clean_whitespace(str(key or ""))
            if not token or value is None:
                continue
            params[token] = str(value)
        return params

    def _resolve_locale(
        self,
        *,
        language: str,
        default_country: str,
    ) -> tuple[str, str, str]:
        token = clean_whitespace(language).replace("_", "-")
        if not token:
            country = default_country or "US"
            return f"en-{country}", "en", country
        parts = [part for part in token.split("-") if part]
        lang = clean_whitespace(parts[0]).lower() or "en"
        region = (
            clean_whitespace(parts[-1]).upper()
            if len(parts) > 1 and len(clean_whitespace(parts[-1])) == 2
            else default_country
        )
        ui_lang = f"{lang}-{region}" if region else lang
        return ui_lang, lang, region

    def _parse_search_results(
        self,
        raw_html: str,
        *,
        base_url: str,
    ) -> tuple[list[SearchProviderResult], list[str]]:
        soup = BeautifulSoup(raw_html, "html.parser")
        results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()
        for block in self._iter_result_blocks(soup):
            for script in block.select("script, style"):
                script.decompose()
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
                    display_url=self._extract_display_url(block),
                    source_engine="google",
                    position=len(results) + 1,
                )
            )
        suggestions = self._extract_suggestions(soup)
        return results, suggestions

    def _iter_result_blocks(self, soup: BeautifulSoup) -> Iterable[Tag]:
        seen: set[int] = set()
        for selector in _GOOGLE_RESULT_SELECTORS:
            for block in soup.select(selector):
                if not isinstance(block, Tag):
                    continue
                marker = id(block)
                if marker in seen:
                    continue
                seen.add(marker)
                yield block

    def _extract_title(self, block: Tag) -> str:
        for selector in ("h3", '[role="heading"]', 'div[role="link"]'):
            node = block.select_one(selector)
            if node is None:
                continue
            title = clean_whitespace(node.get_text(" ", strip=True))
            if title:
                return title
        return ""

    def _extract_snippet(self, block: Tag) -> str:
        for selector in _GOOGLE_SNIPPET_SELECTORS:
            node = block.select_one(selector)
            if node is None:
                continue
            snippet = clean_whitespace(node.get_text(" ", strip=True))
            if snippet:
                return snippet
        return ""

    def _extract_display_url(self, block: Tag) -> str:
        for selector in ("cite", "span.VuuXrf", "div.yuRUbf cite"):
            node = block.select_one(selector)
            if node is None:
                continue
            value = clean_whitespace(node.get_text(" ", strip=True))
            if value:
                return value
        return ""

    def _extract_suggestions(self, soup: BeautifulSoup) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for anchor in soup.select("div.ouy7Mc a"):
            if not isinstance(anchor, Tag):
                continue
            value = clean_whitespace(anchor.get_text(" ", strip=True))
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    def _resolve_result_url(self, raw_href: str, *, base_url: str) -> str:
        href = clean_whitespace(raw_href)
        if not href:
            return ""
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if self._is_google_host(parsed.netloc) and parsed.path == "/url":
            target = parse_qs(parsed.query).get("q", [""])[0]
            href = clean_whitespace(target)
            if not href:
                return ""
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            return ""
        if self._is_google_host(parsed.netloc):
            return ""
        return absolute

    def _raise_if_blocked(self, resp: httpx.Response) -> None:
        parsed = urlparse(str(resp.url))
        text = resp.text.casefold()
        if parsed.netloc == "sorry.google.com" or parsed.path.startswith("/sorry"):
            raise RuntimeError("google blocked the request with a captcha page")
        if any(marker in text for marker in _GOOGLE_BLOCK_MARKERS):
            raise RuntimeError("google blocked the request or requires verification")

    def _coerce_positive_int(self, value: Any) -> int:
        try:
            resolved = int(value)
        except Exception:
            return 10
        return max(1, min(100, resolved))

    def _is_google_host(self, host: str) -> bool:
        normalized = clean_whitespace(host).lower()
        return (
            normalized == "google.com"
            or normalized.endswith(".google.com")
            or ".google." in normalized
        )


__all__ = ["GoogleProvider", "GoogleProviderConfig"]
