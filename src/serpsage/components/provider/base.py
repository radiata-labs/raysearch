from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Any, Generic
from typing_extensions import TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.provider import SearchProviderResult
from serpsage.utils import (
    clean_whitespace,
    iso8601_end_date_exclusive,
    iso8601_start_date_floor,
    normalize_iso8601_string,
    parse_iso8601_datetime,
    published_date_in_range,
)

PROVIDER_ROUTES_TOKEN = "component.provider_routes"  # noqa: S105


class RetrySettings(ComponentConfigBase):
    max_attempts: int = 3
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
        locale: str = "",
        moderation: bool = True,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        normalized_start = self._normalize_published_date_bound(start_published_date)
        normalized_end = self._normalize_published_date_bound(end_published_date)
        normalized_moderation = self._normalize_moderation_flag(moderation)
        return await self._asearch(
            query=query,
            limit=limit,
            locale=locale,
            moderation=normalized_moderation,
            start_published_date=normalized_start,
            end_published_date=normalized_end,
            **kwargs,
        )

    @abstractmethod
    async def _asearch(
        self,
        *,
        query: str,
        limit: int | None = None,
        locale: str = "",
        moderation: bool = True,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        raise NotImplementedError

    def _normalize_moderation_flag(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = clean_whitespace(value).casefold()
            if token in {"false", "0", "off", "no"}:
                return False
            if token in {"true", "1", "on", "yes"}:
                return True
        if isinstance(value, (int, float)):
            return bool(value)
        return True

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
    "PROVIDER_ROUTES_TOKEN",
    "ProviderConfigBase",
    "ProviderMeta",
    "RetrySettings",
    "SearchProviderBase",
]
