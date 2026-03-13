from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Any, Generic, Literal
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
)

GoogleSafeSearchKey = Literal["off", "medium", "high"]
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
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        normalized_start = self._normalize_published_date_bound(start_published_date)
        normalized_end = self._normalize_published_date_bound(end_published_date)
        return await self._asearch(
            query=query,
            limit=limit,
            locale=locale,
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
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        raise NotImplementedError

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


__all__ = [
    "GoogleSafeSearchKey",
    "PROVIDER_ROUTES_TOKEN",
    "ProviderConfigBase",
    "ProviderMeta",
    "RetrySettings",
    "SearchProviderBase",
]
