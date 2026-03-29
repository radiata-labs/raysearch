# SPDX-License-Identifier: AGPL-3.0-or-later
"""This is the implementation of the Google News engine.

Google News has a different region handling compared to Google WEB.

- the ``ceid`` argument has to be set (see ``_GOOGLE_NEWS_CEID_LIST``)
- the ``hl`` argument has to be set correctly
- the ``gl`` argument is mandatory

If one of these arguments is not set correctly, the request is redirected to
the Google consent dialog.

Google News also differs from Google WEB in one important way: it returns all
results for a query on the first page, so the provider does not implement
paging.

Configuration
=============

Example configuration in this project:

.. code:: yaml

   google_news:
     enabled: true
     base_url: https://news.google.com/search

Notes
=====

- The provider follows the same ``ceid``-driven request construction used by
  the SearXNG Google News engine.
- Google News result links are internal article tokens that must be decoded
  back into their target article URLs.
- The feed is article-oriented and does not provide reliable machine-readable
  publication timestamps for every hit.
"""

from __future__ import annotations

import base64
from typing import Any
from typing_extensions import override
from urllib.parse import urlencode, urljoin, urlparse

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

_DEFAULT_GOOGLE_NEWS_BASE_URL = "https://news.google.com/search"
_DEFAULT_GOOGLE_NEWS_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 11; Pixel 5 Build/RQ3A.210905.001; wv) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 "
    "Chrome/120.0.0.0 Mobile Safari/537.36 GSA/14.48.26.29.arm64"
)
_GOOGLE_NEWS_BLOCK_MARKERS = (
    "unusual traffic",
    "sorry.google.com",
    "/sorry/",
    "detected unusual traffic",
    "our systems have detected",
    "consent.google.com",
)
_GOOGLE_NEWS_CEID_LIST = (
    "AE:ar",
    "AR:es-419",
    "AT:de",
    "AU:en",
    "BD:bn",
    "BE:fr",
    "BE:nl",
    "BG:bg",
    "BR:pt-419",
    "BW:en",
    "CA:en",
    "CA:fr",
    "CH:de",
    "CH:fr",
    "CL:es-419",
    "CN:zh-Hans",
    "CO:es-419",
    "CU:es-419",
    "CZ:cs",
    "DE:de",
    "EG:ar",
    "ES:es",
    "ET:en",
    "FR:fr",
    "GB:en",
    "GH:en",
    "GR:el",
    "HK:zh-Hant",
    "HU:hu",
    "ID:en",
    "ID:id",
    "IE:en",
    "IL:en",
    "IL:he",
    "IN:bn",
    "IN:en",
    "IN:hi",
    "IN:ml",
    "IN:mr",
    "IN:ta",
    "IN:te",
    "IT:it",
    "JP:ja",
    "KE:en",
    "KR:ko",
    "LB:ar",
    "LT:lt",
    "LV:en",
    "LV:lv",
    "MA:fr",
    "MX:es-419",
    "MY:en",
    "NA:en",
    "NG:en",
    "NL:nl",
    "NO:no",
    "NZ:en",
    "PE:es-419",
    "PH:en",
    "PK:en",
    "PL:pl",
    "PT:pt-150",
    "RO:ro",
    "RS:sr",
    "RU:ru",
    "SA:ar",
    "SE:sv",
    "SG:en",
    "SI:sl",
    "SK:sk",
    "SN:fr",
    "TH:th",
    "TR:tr",
    "TW:zh-Hant",
    "TZ:en",
    "UA:ru",
    "UA:uk",
    "UG:en",
    "US:en",
    "US:es-419",
    "VE:es-419",
    "VN:vi",
    "ZA:en",
    "ZW:en",
)
_GOOGLE_NEWS_SKIP_CEID = {"ET:en", "ID:en", "LV:en"}
_GOOGLE_NEWS_LANGUAGE_OVERRIDES = {"NO:no": "nb-NO"}


class GoogleNewsProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "google_news"

    base_url: str = _DEFAULT_GOOGLE_NEWS_BASE_URL
    user_agent: str = _DEFAULT_GOOGLE_NEWS_USER_AGENT
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
        if env.get("GOOGLE_NEWS_BASE_URL"):
            payload["base_url"] = env["GOOGLE_NEWS_BASE_URL"]
        return payload


class GoogleNewsProvider(
    SearchProviderBase[GoogleNewsProviderConfig],
    meta=ProviderMeta(
        name="google_news",
        website="https://news.google.com/",
        description="News search through Google News with region-aware query parameters and article-oriented result parsing.",
        preference="Prefer current-events, company-news, policy, product launch, and publication-oriented searches where news vertical ranking matters.",
        categories=["news"],
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
        **_kwargs: Any,
    ) -> list[SearchProviderResult]:
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        request = self._build_request(
            query=normalized_query,
            language=language,
            location=location,
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
        results = self._parse_results(resp.text, base_url=str(resp.url))
        return self._trim_results(results, limit=limit)

    def _build_request(
        self,
        *,
        query: str,
        language: str,
        location: str,
    ) -> dict[str, Any]:
        ceid = self._resolve_ceid(language=language, location=location)
        region, language = ceid.split(":", 1)
        hl = self._resolve_hl(region=region, language=language)
        params = {
            "q": query,
            "hl": hl,
            "lr": f"lang_{language.split('-', 1)[0]}",
            "gl": region,
        }
        query_url = (
            clean_whitespace(str(self.config.base_url or _DEFAULT_GOOGLE_NEWS_BASE_URL))
            + "?"
            + urlencode(params)
            + f"&ceid={ceid}"
        )
        return {
            "url": query_url,
            "headers": self._build_headers(),
            "cookies": self._build_cookies(),
        }

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers.setdefault(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        headers["User-Agent"] = (
            clean_whitespace(str(self.config.user_agent or ""))
            or _DEFAULT_GOOGLE_NEWS_USER_AGENT
        )
        return headers

    def _build_cookies(self) -> dict[str, str]:
        cookies = dict(self.config.cookies or {})
        cookies["CONSENT"] = "YES+"
        return cookies

    def _parse_results(
        self,
        raw_html: str,
        *,
        base_url: str,
    ) -> list[SearchProviderResult]:
        soup = BeautifulSoup(raw_html, "html.parser")
        results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()
        for block in soup.select("div.xrnccd"):
            if not isinstance(block, Tag):
                continue
            article = block.select_one("article")
            if article is None:
                continue
            anchor = article.select_one("a[href]")
            if anchor is None:
                continue
            resolved_url = self._decode_result_url(
                str(anchor.get("href", "")),
                base_url=base_url,
            )
            if not resolved_url:
                continue
            key = resolved_url.casefold()
            if key in seen_urls:
                continue
            title = clean_whitespace(
                (article.select_one("h3") or article).get_text(" ", strip=True)
            )
            source_node = article.select_one("a[data-n-tid], div.QmrVtf a")
            source = clean_whitespace(
                source_node.get_text(" ", strip=True) if source_node is not None else ""
            )
            time_node = article.select_one("time")
            published_label = clean_whitespace(
                time_node.get_text(" ", strip=True) if time_node is not None else ""
            )
            snippet = clean_whitespace(
                " / ".join(
                    token
                    for token in (source, published_label)
                    if clean_whitespace(token)
                )
            )
            if not title:
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

    def _decode_result_url(self, raw_href: str, *, base_url: str) -> str:
        href = clean_whitespace(raw_href)
        if not href:
            return ""
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme in {"http", "https"} and not self._is_google_host(
            parsed.netloc
        ):
            return absolute
        encoded_segment = ""
        for part in reversed(
            [segment for segment in parsed.path.split("/") if segment]
        ):
            if part not in {"articles", "read"}:
                encoded_segment = part
                break
        if not encoded_segment:
            return ""
        padded = encoded_segment + "=" * ((4 - len(encoded_segment) % 4) % 4)
        try:
            decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        except Exception:
            return ""
        start_index = decoded.find(b"http")
        if start_index < 0:
            return ""
        payload = decoded[start_index:]
        for delimiter in (b"\xd2", b"\x00", b"&ved=", b"\x08\x13"):
            index = payload.find(delimiter)
            if index > 0:
                payload = payload[:index]
                break
        try:
            resolved = payload.decode("utf-8", errors="ignore")
        except Exception:
            return ""
        resolved = clean_whitespace(resolved)
        if not resolved:
            return ""
        parsed_resolved = urlparse(resolved)
        if parsed_resolved.scheme not in {"http", "https"}:
            return ""
        if self._is_google_host(parsed_resolved.netloc):
            return ""
        return resolved

    def _resolve_ceid(self, *, language: str, location: str) -> str:
        ceid_map = _google_news_ceid_map()
        normalized = language
        normalized_location = location
        base_language = ""
        region = normalized_location
        if normalized:
            lowered = normalized.casefold()
            direct = ceid_map.get(lowered, "")
            if direct and (
                not normalized_location
                or direct.split(":", 1)[0] == normalized_location
            ):
                return direct
            base_language = clean_whitespace(normalized.split("-", 1)[0]).lower()
            if not region:
                parts = [part for part in normalized.split("-") if part]
                if len(parts) >= 2 and self._is_region_subtag(parts[-1]):
                    region = clean_whitespace(parts[-1]).upper()
        if region:
            if base_language:
                direct = f"{region}:{base_language}"
                if (
                    direct in _GOOGLE_NEWS_CEID_LIST
                    and direct not in _GOOGLE_NEWS_SKIP_CEID
                ):
                    return direct
            for ceid in _GOOGLE_NEWS_CEID_LIST:
                if ceid.startswith(f"{region}:") and ceid not in _GOOGLE_NEWS_SKIP_CEID:
                    return ceid
        if not base_language and normalized:
            lowered = normalized.casefold()
            if lowered in ceid_map:
                return ceid_map[lowered]
        if normalized:
            for ceid in _GOOGLE_NEWS_CEID_LIST:
                if ceid.split(":", 1)[1].casefold() == normalized.casefold():
                    return ceid
        for ceid in _GOOGLE_NEWS_CEID_LIST:
            if (
                ceid.split(":", 1)[1].split("-", 1)[0] == base_language
                and ceid not in _GOOGLE_NEWS_SKIP_CEID
            ):
                return ceid
        return "US:en"

    def _is_region_subtag(self, value: str) -> bool:
        token = clean_whitespace(value)
        return (len(token) == 2 and token.isalpha()) or (
            len(token) == 3 and token.isdigit()
        )

    def _resolve_hl(self, *, region: str, language: str) -> str:
        lang_parts = [part for part in language.split("-") if part]
        base_language = lang_parts[0]
        suffix = lang_parts[1] if len(lang_parts) > 1 else ""
        if suffix and suffix not in {"Hans", "Hant"}:
            if region.lower() == base_language:
                return f"{base_language}-{region}"
            return f"{base_language}-{suffix}"
        if region.lower() != base_language:
            if region in {"AT", "BE", "CH", "IL", "SA", "IN", "BD", "PT"}:
                return base_language
            return f"{base_language}-{region}"
        return base_language

    def _raise_if_blocked(self, resp: httpx.Response) -> None:
        parsed = urlparse(str(resp.url))
        text = resp.text.casefold()
        if parsed.netloc == "sorry.google.com" or parsed.path.startswith("/sorry"):
            raise RuntimeError("google news blocked the request with a captcha page")
        if parsed.netloc == "consent.google.com":
            raise RuntimeError("google news redirected the request to the consent page")
        if any(marker in text for marker in _GOOGLE_NEWS_BLOCK_MARKERS):
            raise RuntimeError(
                "google news blocked the request or requires verification"
            )

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
            normalized in {"google.com", "news.google.com"}
            or normalized.endswith((".google.com", ".news.google.com"))
            or ".google." in normalized
        )


def _google_news_ceid_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for ceid in _GOOGLE_NEWS_CEID_LIST:
        if ceid in _GOOGLE_NEWS_SKIP_CEID:
            continue
        region, language = ceid.split(":", 1)
        language_tag = _GOOGLE_NEWS_LANGUAGE_OVERRIDES.get(
            ceid,
            f"{language}-{region}",
        )
        out[language_tag.casefold()] = ceid
        out[f"{language.casefold()}-{region.casefold()}"] = ceid
        out[f"{language.split('-', 1)[0].casefold()}-{region.casefold()}"] = ceid
    return out


__all__ = ["GoogleNewsProvider", "GoogleNewsProviderConfig"]
