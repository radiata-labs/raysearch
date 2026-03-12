from __future__ import annotations

import re
from typing import Any
from typing_extensions import override
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import Field, field_validator

from serpsage.components.http.base import HttpClientBase
from serpsage.components.provider.base import (
    GoogleSafeSearchKey,
    ProviderConfigBase,
    SearchProviderBase,
)
from serpsage.dependencies import Depends
from serpsage.models.components.provider import (
    SearchProviderResponse,
    SearchProviderResult,
)
from serpsage.utils import clean_whitespace

_GOOGLE_TIME_RANGE_MAP = {"day": "d", "week": "w", "month": "m", "year": "y"}
_GOOGLE_BLOCK_MARKERS = (
    "unusual traffic",
    "sorry.google.com",
    "/sorry/",
    "detected unusual traffic",
    "our systems have detected",
)
_GOOGLE_RESULT_SELECTORS = ("div.MjjYud", "div.g")
_GOOGLE_SNIPPET_SELECTORS = ('[data-sncf="1"]',)
_GOOGLE_SUGGESTION_SELECTOR = "div.ouy7Mc a"
_DEFAULT_GOOGLE_BASE_URL = "https://www.google.com/search"
_DEFAULT_GOOGLE_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 11; Pixel 5 Build/RQ3A.210905.001; wv) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 "
    "Chrome/120.0.0.0 Mobile Safari/537.36 GSA/14.48.26.29.arm64"
)
_RE_DATA_IMAGE = re.compile(r'"(dimg_[^"]*)"[^;]*;(data:image[^;]*;[^;]*);')
_RE_DATA_IMAGE_END = re.compile(r'"(dimg_[^"]*)"[^;]*;(data:image[^;]*;[^;]*)$')


class GoogleProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "google"

    base_url: str = _DEFAULT_GOOGLE_BASE_URL
    user_agent: str = _DEFAULT_GOOGLE_USER_AGENT
    safe: GoogleSafeSearchKey = "off"
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


class GoogleProvider(SearchProviderBase[GoogleProviderConfig]):
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
        normalized_query = clean_whitespace(query)
        normalized_language = clean_whitespace(language)
        if not normalized_query:
            raise ValueError("query must not be empty")

        request = self._build_request(
            query=normalized_query,
            page=page,
            language=normalized_language,
            kwargs=kwargs,
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
        results, suggestions = self._parse_search_results(
            resp.text,
            base_url=str(resp.url),
        )
        return SearchProviderResponse(
            provider_backend="google",
            query=normalized_query,
            page=int(page),
            language=normalized_language,
            suggestions=suggestions,
            results=results,
            metadata={
                "request_url": str(resp.url),
                "google_domain": str(request["base_url"]),
            },
        )

    def _build_request(
        self,
        *,
        query: str,
        page: int,
        language: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        cfg = self.config
        extra = dict(kwargs)
        safe = clean_whitespace(str(extra.pop("safe", cfg.safe) or "")).casefold()
        time_range = clean_whitespace(str(extra.pop("time_range", "") or "")).casefold()
        country = clean_whitespace(str(extra.pop("country", cfg.country) or "")).upper()
        google_info = self._get_google_info(
            language=language,
            default_country=country or "US",
        )
        start = max(0, (int(page) - 1) * 10)
        query_params: dict[str, str | int] = {
            "q": query,
            **google_info["params"],
            "filter": "0",
            "start": start,
        }
        for key, value in extra.items():
            token = clean_whitespace(str(key or ""))
            if not token or value is None:
                continue
            query_params[token] = str(value)
        query_url = str(google_info["base_url"]) + "?" + urlencode(query_params)
        if time_range in _GOOGLE_TIME_RANGE_MAP:
            query_url += "&" + urlencode(
                {"tbs": "qdr:" + _GOOGLE_TIME_RANGE_MAP[time_range]}
            )
        if safe in {"medium", "high"}:
            query_url += "&" + urlencode({"safe": safe})
        return {
            "url": query_url,
            "headers": google_info["headers"],
            "cookies": google_info["cookies"],
            "base_url": google_info["base_url"],
        }

    def _get_google_info(
        self,
        *,
        language: str,
        default_country: str,
    ) -> dict[str, Any]:
        cfg = self.config
        locale = clean_whitespace(language).replace("_", "-")
        region_in_locale = False
        if not locale or locale == "all":
            locale = "en"
        parts = [part for part in locale.split("-") if part]
        lang = clean_whitespace(parts[0]).lower() or "en"
        country = default_country or "US"
        if len(parts) > 1 and len(clean_whitespace(parts[-1])) == 2:
            country = clean_whitespace(parts[-1]).upper()
            region_in_locale = True
        lr = f"lang_{lang}"
        if region_in_locale and lang == "zh":
            lr = f"lang_{lang}-{country}"
        if clean_whitespace(language).casefold() == "all":
            lr = ""
        lang_code = lr.split("_")[-1] if lr else lang
        base_url = clean_whitespace(str(cfg.base_url or _DEFAULT_GOOGLE_BASE_URL))
        headers = dict(cfg.headers or {})
        headers["Accept"] = "*/*"
        headers["User-Agent"] = self._google_user_agent()
        cookies = dict(cfg.cookies or {})
        cookies["CONSENT"] = "YES+"
        params = {
            "hl": f"{lang_code}-{country}",
            "lr": lr,
            "cr": f"country{country}" if region_in_locale else "",
            "ie": "utf8",
            "oe": "utf8",
        }
        return {
            "base_url": base_url,
            "params": params,
            "headers": headers,
            "cookies": cookies,
            "country": country,
            "language": lr,
        }

    def _parse_search_results(
        self,
        raw_html: str,
        *,
        base_url: str,
    ) -> tuple[list[SearchProviderResult], list[str]]:
        soup = BeautifulSoup(raw_html, "html.parser")
        data_image_map = self._parse_data_images(raw_html)
        results: list[SearchProviderResult] = []
        suggestions: list[str] = []
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
            snippet = self._extract_snippet(block)
            display_url = self._extract_display_url(block)
            metadata: dict[str, Any] = {}
            thumbnail = self._extract_thumbnail(block, data_image_map)
            if thumbnail:
                metadata["thumbnail"] = thumbnail
            seen_urls.add(key)
            results.append(
                SearchProviderResult(
                    url=resolved_url,
                    title=title,
                    snippet=snippet,
                    display_url=display_url,
                    source_engine="google",
                    position=len(results) + 1,
                    metadata=metadata,
                )
            )
        for anchor in soup.select(_GOOGLE_SUGGESTION_SELECTOR):
            if not isinstance(anchor, Tag):
                continue
            suggestion = clean_whitespace(anchor.get_text(" ", strip=True))
            if suggestion:
                suggestions.append(suggestion)
        return results, self._dedupe_text(suggestions)

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

    def _extract_display_url(self, block: Tag) -> str:
        for selector in ("cite", "span.VuuXrf", "div.yuRUbf cite"):
            node = block.select_one(selector)
            if node is None:
                continue
            value = clean_whitespace(node.get_text(" ", strip=True))
            if value:
                return value
        return ""

    def _extract_thumbnail(
        self,
        block: Tag,
        data_image_map: dict[str, str],
    ) -> str:
        image = block.select_one("img[src]")
        if image is None:
            return ""
        src = clean_whitespace(str(image.get("src", "")))
        if src.startswith("data:image"):
            img_id = clean_whitespace(str(image.get("id", "")))
            if img_id:
                return clean_whitespace(data_image_map.get(img_id, src))
        return src

    def _parse_data_images(self, text: str) -> dict[str, str]:
        data_image_map: dict[str, str] = {}
        for img_id, data_image in _RE_DATA_IMAGE.findall(text):
            end_pos = data_image.rfind("=")
            if end_pos > 0:
                data_image = data_image[: end_pos + 1]
            data_image_map[img_id] = data_image
        last = _RE_DATA_IMAGE_END.search(text)
        if last is not None:
            data_image_map[last.group(1)] = last.group(2)
        return data_image_map

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

    def _google_user_agent(self) -> str:
        configured = clean_whitespace(str(self.config.user_agent or ""))
        return configured or _DEFAULT_GOOGLE_USER_AGENT

    def _dedupe_text(self, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = clean_whitespace(value)
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _is_google_host(self, host: str) -> bool:
        normalized = clean_whitespace(host).lower()
        return (
            normalized == "google.com"
            or normalized.endswith(".google.com")
            or ".google." in normalized
        )


__all__ = ["GoogleProvider", "GoogleProviderConfig"]
