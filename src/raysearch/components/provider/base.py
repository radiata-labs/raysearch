from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Any, Generic
from typing_extensions import TypeVar

import anyio
from pydantic import BaseModel, Field, field_validator, model_validator

from raysearch.components.base import ComponentBase, ComponentConfigBase
from raysearch.models.components.provider import SearchProviderResult
from raysearch.utils import (
    clean_whitespace,
    iso8601_end_date_exclusive,
    iso8601_start_date_floor,
    normalize_iso8601_string,
    parse_iso8601_datetime,
    published_date_in_range,
)

_ISO_COUNTRY_CODE_RE = re.compile(r"^[A-Za-z]{2}$")
_LANGUAGE_TAG_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")


class RetrySettings(ComponentConfigBase):
    retries: int = 1
    delay_ms: int = 200


class ProviderConfigBase(ComponentConfigBase):
    base_url: str = ""
    timeout_s: float = 20.0
    allow_redirects: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)
    retry: RetrySettings = Field(default_factory=RetrySettings)

    @property
    def name(self) -> str:
        return self.__setting_name__

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if normalized and not normalized.startswith(("http://", "https://")):
            raise ValueError("provider base_url must start with http:// or https://")
        return normalized

    @model_validator(mode="after")
    def _validate_provider(self) -> ProviderConfigBase:
        if float(self.timeout_s) <= 0:
            raise ValueError("provider timeout_s must be > 0")
        return self


class ProviderMeta(BaseModel):
    name: str
    website: str
    description: str
    preference: str
    categories: list[str]


ProviderConfigT = TypeVar(
    "ProviderConfigT",
    bound=ProviderConfigBase,
    default=ProviderConfigBase,
)


class SearchProviderBase(ComponentBase[ProviderConfigT], ABC, Generic[ProviderConfigT]):
    meta: ProviderMeta

    def __init_subclass__(
        cls,
        meta: ProviderMeta | None = None,
        config: type[ProviderConfigT] | None = None,
        **_kwargs: Any,
    ) -> None:
        if meta is not None:
            cls.meta = meta
        return super().__init_subclass__(config, **_kwargs)

    async def asearch(
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
        normalized_start = self._normalize_published_date_bound(start_published_date)
        normalized_end = self._normalize_published_date_bound(end_published_date)
        normalized_language = self._normalize_language(language)
        normalized_location = self._normalize_location(location)
        retry = self.config.retry
        attempts = max(1, int(retry.retries) + 1)
        delay_s = max(0.0, float(getattr(retry, "delay_ms", 0.0)) / 1000.0)
        for attempt_index in range(attempts):
            try:
                return await self._asearch(
                    query=query,
                    limit=limit,
                    language=normalized_language,
                    location=normalized_location,
                    moderation=moderation,
                    start_published_date=normalized_start,
                    end_published_date=normalized_end,
                    **kwargs,
                )
            except Exception:
                if attempt_index >= attempts - 1:
                    raise
                if delay_s > 0:
                    await anyio.sleep(delay_s)
        raise RuntimeError("search retry loop exhausted without result")

    @abstractmethod
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
        raise NotImplementedError

    def _normalize_language(self, value: str) -> str:
        token = clean_whitespace(str(value or "")).replace("_", "-")
        if not token or token.casefold() == "all":
            return ""
        if not _LANGUAGE_TAG_RE.fullmatch(token):
            raise ValueError("language must be a valid BCP 47 language tag")
        parts = [part for part in token.split("-") if part]
        if not parts:
            return ""
        normalized_parts = [parts[0].lower()]
        for part in parts[1:]:
            if len(part) == 4 and part.isalpha():
                normalized_parts.append(part.title())
                continue
            if (len(part) == 2 and part.isalpha()) or (
                len(part) == 3 and part.isdigit()
            ):
                normalized_parts.append(part.upper())
                continue
            normalized_parts.append(part.lower())
        return "-".join(normalized_parts)

    def _normalize_location(self, value: str) -> str:
        token = clean_whitespace(str(value or "")).upper()
        if not token:
            return ""
        if not _ISO_COUNTRY_CODE_RE.fullmatch(token):
            raise ValueError("location must be a two-letter ISO country code")
        return token

    def _normalize_published_date_bound(self, value: str | None) -> str:
        token = clean_whitespace(str(value or ""))
        if not token:
            return ""
        return normalize_iso8601_string(token)

    def _coarse_published_date_bounds(
        self,
        *,
        start_published_date: str | None,
        end_published_date: str | None,
    ) -> tuple[str, str]:
        start_value = self._normalize_published_date_bound(start_published_date)
        end_value = self._normalize_published_date_bound(end_published_date)
        if not start_value and not end_value:
            return "", ""
        return (
            iso8601_start_date_floor(start_value),
            iso8601_end_date_exclusive(end_value),
        )

    def _relative_time_range_from_bounds(
        self,
        *,
        start_published_date: str | None,
        end_published_date: str | None,
    ) -> str:
        start_value = clean_whitespace(str(start_published_date or ""))
        if not start_value:
            return ""
        start_at = parse_iso8601_datetime(start_value)
        if start_at is None:
            return ""
        end_value = clean_whitespace(str(end_published_date or ""))
        end_at = parse_iso8601_datetime(end_value) if end_value else None
        now_utc = datetime.now(UTC)
        effective_end = end_at or now_utc
        if effective_end < start_at:
            return ""
        windows = (
            ("day", timedelta(days=1)),
            ("week", timedelta(days=7)),
            ("month", timedelta(days=31)),
            ("year", timedelta(days=366)),
        )
        for name, width in windows:
            window_start = now_utc - width
            if start_at >= window_start and effective_end >= window_start:
                return name
        return ""

    def _filter_results_by_published_date(
        self,
        *,
        results: list[SearchProviderResult],
        start_published_date: str | None,
        end_published_date: str | None,
        include_undated: bool = False,
    ) -> list[SearchProviderResult]:
        start_value = self._normalize_published_date_bound(start_published_date)
        end_value = self._normalize_published_date_bound(end_published_date)
        if not start_value and not end_value:
            return results
        filtered: list[SearchProviderResult] = []
        for item in results:
            published_date = self._normalize_published_date_bound(item.published_date)
            if not published_date:
                if include_undated:
                    filtered.append(item)
                continue
            if not published_date_in_range(
                published_date,
                start_published_date=start_value,
                end_published_date=end_value,
            ):
                continue
            if published_date == item.published_date:
                filtered.append(item)
                continue
            filtered.append(item.model_copy(update={"published_date": published_date}))
        return filtered

    def _initial_dated_fetch_limit(
        self,
        *,
        target_limit: int,
        max_limit: int,
        multiplier: int = 2,
    ) -> int:
        target = max(1, int(target_limit))
        cap = max(1, int(max_limit))
        factor = max(1, int(multiplier))
        return min(cap, max(target, target * factor))


__all__ = [
    "ProviderConfigBase",
    "ProviderMeta",
    "RetrySettings",
    "SearchProviderBase",
]
