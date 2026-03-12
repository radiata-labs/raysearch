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
from serpsage.utils import clean_whitespace

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
        locale: str = "",
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        cfg = self.config
        if not query or not str(query).strip():
            raise ValueError("query must not be empty")
        payload: dict[str, str] = {"q": str(query), "format": "json"}
        page_size = max(1, int(limit)) if limit is not None else None
        if page_size is not None:
            payload["limit"] = str(page_size)
        if locale:
            payload["language"] = str(locale)
        self._merge_extra_payload(payload=payload, extra=kwargs)
        headers = dict(cfg.headers or {})
        if cfg.user_agent:
            headers.setdefault("User-Agent", str(cfg.user_agent))
        if cfg.api_key:
            headers.setdefault("Authorization", f"Bearer {cfg.api_key}")
        resp = await self.http.client.get(
            cfg.base_url,
            params=payload,
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s),
            follow_redirects=bool(cfg.allow_redirects),
        )
        resp.raise_for_status()
        data = resp.json()
        raw_results = data.get("results", [])
        results: list[SearchProviderResult] = []
        if isinstance(raw_results, list):
            for raw in raw_results:
                if not isinstance(raw, dict):
                    continue
                url = clean_whitespace(str(raw.get("url") or ""))
                if not url:
                    continue
                snippet = raw.get("content")
                if snippet is None:
                    snippet = raw.get("description")
                _ = (
                    raw.get("pretty_url"),
                    raw.get("engine"),
                    {
                        key: value
                        for key, value in raw.items()
                        if key
                        not in {"url", "title", "content", "description", "engine"}
                    },
                )
                results.append(
                    SearchProviderResult(
                        url=url,
                        title=str(raw.get("title") or ""),
                        snippet=str(snippet or ""),
                        engine=self.config.name,
                    )
                )
        _ = (
            data.get("number_of_results"),
            data.get("suggestions"),
            self._coerce_page(payload.get("pageno")),
            locale,
            {
                key: value
                for key, value in data.items()
                if key not in {"results", "number_of_results", "suggestions"}
            },
        )
        return results[:page_size] if page_size is not None else results

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

    def _coerce_page(self, value: Any) -> int:
        try:
            page = int(value)
        except Exception:
            return 1
        return max(1, page)


__all__ = ["SearxngProvider", "SearxngProviderConfig"]
