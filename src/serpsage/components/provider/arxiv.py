from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from typing_extensions import override
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import field_validator

from serpsage.components.http.base import HttpClientBase
from serpsage.components.provider.base import ProviderConfigBase, SearchProviderBase
from serpsage.dependencies import Depends
from serpsage.models.components.provider import (
    SearchProviderResponse,
    SearchProviderResult,
)
from serpsage.utils import clean_whitespace

_DEFAULT_ARXIV_BASE_URL = "https://export.arxiv.org/api/query"
_DEFAULT_ARXIV_USER_AGENT = "serpsage-arxiv-provider/1.0"
_DEFAULT_ARXIV_RESULTS_PER_PAGE = 10
_MAX_ARXIV_RESULTS_PER_PAGE = 100


class ArxivProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "arxiv"

    base_url: str = _DEFAULT_ARXIV_BASE_URL
    user_agent: str = _DEFAULT_ARXIV_USER_AGENT
    search_prefix: str = "all"
    results_per_page: int = _DEFAULT_ARXIV_RESULTS_PER_PAGE

    @field_validator("user_agent", "search_prefix")
    @classmethod
    def _normalize_text_fields(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("arxiv results_per_page must be > 0")
        if size > _MAX_ARXIV_RESULTS_PER_PAGE:
            raise ValueError(
                f"arxiv results_per_page must be <= {_MAX_ARXIV_RESULTS_PER_PAGE}"
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
        if env.get("ARXIV_BASE_URL"):
            payload["base_url"] = env["ARXIV_BASE_URL"]
        return payload


class ArxivProvider(SearchProviderBase[ArxivProviderConfig]):
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
        search_prefix = (
            clean_whitespace(str(extra.pop("search_prefix", cfg.search_prefix) or ""))
            or "all"
        )
        per_page = self._coerce_per_page(
            extra.pop("max_results", extra.pop("per_page", cfg.results_per_page))
        )
        params = self._build_params(
            query=normalized_query,
            search_prefix=search_prefix,
            page=page,
            per_page=per_page,
            extra=extra,
        )
        headers = dict(cfg.headers or {})
        headers["Accept"] = "application/atom+xml, application/xml;q=0.9, */*;q=0.8"
        headers["User-Agent"] = str(cfg.user_agent or _DEFAULT_ARXIV_USER_AGENT)
        resp = await self.http.client.get(
            str(cfg.base_url),
            params=params,
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s),
            follow_redirects=bool(cfg.allow_redirects),
        )
        resp.raise_for_status()
        total_results, results = self._parse_feed(resp.content)
        return SearchProviderResponse(
            provider_backend="arxiv",
            query=normalized_query,
            page=int(page),
            language=normalized_language,
            total_results=total_results,
            results=results,
            metadata={
                "request_url": str(resp.url),
                "arxiv_domain": str(cfg.base_url),
                "search_prefix": search_prefix,
            },
        )

    def _build_params(
        self,
        *,
        query: str,
        search_prefix: str,
        page: int,
        per_page: int,
        extra: dict[str, Any],
    ) -> dict[str, str]:
        params: dict[str, str] = {
            "search_query": f"{search_prefix}:{query}",
            "start": str(max(0, (int(page) - 1) * per_page)),
            "max_results": str(per_page),
        }
        for key, value in extra.items():
            token = clean_whitespace(str(key or ""))
            if not token or value is None:
                continue
            params[token] = str(value)
        return params

    def _parse_feed(
        self, content: bytes
    ) -> tuple[int | None, list[SearchProviderResult]]:
        soup = BeautifulSoup(content, "xml")
        total_results = self._parse_total_results(soup)
        results: list[SearchProviderResult] = []
        for index, entry in enumerate(soup.find_all("entry"), start=1):
            if not isinstance(entry, Tag):
                continue
            item = self._parse_entry(entry, position=index)
            if item is None:
                continue
            results.append(item)
        return total_results, results

    def _parse_total_results(self, soup: BeautifulSoup) -> int | None:
        node = self._find_first(soup, "opensearch:totalResults", "totalResults")
        if node is None:
            return None
        try:
            return int(clean_whitespace(node.get_text(" ", strip=True)))
        except ValueError:
            return None

    def _parse_entry(self, entry: Tag, *, position: int) -> SearchProviderResult | None:
        title = self._text(self._find_first(entry, "title"))
        url = self._text(self._find_first(entry, "id"))
        if not title or not url:
            return None
        summary = self._text(self._find_first(entry, "summary"))
        authors = [
            value
            for value in (
                self._text(self._find_first(node, "name"))
                for node in entry.find_all("author")
            )
            if value
        ]
        doi = self._text(self._find_first(entry, "arxiv:doi", "doi"))
        journal = self._text(
            self._find_first(entry, "arxiv:journal_ref", "journal_ref")
        )
        comments = self._text(self._find_first(entry, "arxiv:comment", "comment"))
        published = self._parse_published(
            self._text(self._find_first(entry, "published"))
        )
        tags = [
            clean_whitespace(str(node.get("term") or ""))
            for node in entry.find_all("category")
            if isinstance(node, Tag) and clean_whitespace(str(node.get("term") or ""))
        ]
        pdf_url = ""
        for node in entry.find_all("link"):
            if not isinstance(node, Tag):
                continue
            if clean_whitespace(str(node.get("title") or "")) != "pdf":
                continue
            pdf_url = clean_whitespace(str(node.get("href") or ""))
            if pdf_url:
                break
        metadata: dict[str, Any] = {}
        if published:
            metadata["published_date"] = published
        if doi:
            metadata["doi"] = doi
        if authors:
            metadata["authors"] = authors
        if journal:
            metadata["journal"] = journal
        if tags:
            metadata["tags"] = tags
        if comments:
            metadata["comments"] = comments
        if pdf_url:
            metadata["pdf_url"] = pdf_url
        return SearchProviderResult(
            url=url,
            title=title,
            snippet=summary,
            display_url=self._display_url(url),
            source_engine="arxiv",
            position=position,
            metadata=metadata,
        )

    def _find_first(self, node: BeautifulSoup | Tag, *names: str) -> Tag | None:
        for name in names:
            found = node.find(name)
            if isinstance(found, Tag):
                return found
        return None

    def _text(self, node: Tag | None) -> str:
        if node is None:
            return ""
        return clean_whitespace(node.get_text(" ", strip=True))

    def _parse_published(self, value: str) -> str:
        token = clean_whitespace(value)
        if not token:
            return ""
        try:
            parsed = datetime.fromisoformat(token)
        except ValueError:
            return token
        return parsed.astimezone(UTC).isoformat()

    def _display_url(self, url: str) -> str:
        parsed = urlparse(url)
        host = clean_whitespace(parsed.netloc)
        path = clean_whitespace(parsed.path)
        return clean_whitespace(f"{host}{path}")

    def _coerce_per_page(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(_MAX_ARXIV_RESULTS_PER_PAGE, size))


__all__ = ["ArxivProvider", "ArxivProviderConfig"]
