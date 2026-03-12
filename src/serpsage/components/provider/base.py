from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal
from typing_extensions import TypeVar

from pydantic import Field, field_validator, model_validator

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.provider import SearchProviderResult

GoogleSafeSearchKey = Literal["off", "medium", "high"]


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


ProviderConfigT = TypeVar(
    "ProviderConfigT",
    bound=ProviderConfigBase,
    default=ProviderConfigBase,
)


class SearchProviderBase(ComponentBase[ProviderConfigT], ABC, Generic[ProviderConfigT]):
    async def asearch(
        self,
        *,
        query: str,
        limit: int | None = None,
        locale: str = "",
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        return await self._asearch(query=query, limit=limit, locale=locale, **kwargs)

    @abstractmethod
    async def _asearch(
        self,
        *,
        query: str,
        limit: int | None = None,
        locale: str = "",
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        raise NotImplementedError


__all__ = [
    "GoogleSafeSearchKey",
    "ProviderConfigBase",
    "RetrySettings",
    "SearchProviderBase",
]
