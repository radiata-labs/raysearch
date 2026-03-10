from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal
from typing_extensions import TypeVar

from pydantic import Field, field_validator, model_validator

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.provider import SearchProviderResponse

GoogleSafeSearchKey = Literal["off", "medium", "high"]


class RetrySettings(ComponentConfigBase):
    max_attempts: int = 3
    delay_ms: int = 200


class ProviderConfigBase(ComponentConfigBase):
    base_url: str = ""
    api_key: str | None = None
    timeout_s: float = 20.0
    allow_redirects: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)
    user_agent: str = ""
    results_per_page: int = 10
    retry: RetrySettings = Field(default_factory=RetrySettings)

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if normalized and not normalized.startswith(("http://", "https://")):
            raise ValueError("provider base_url must start with http:// or https://")
        return normalized

    @field_validator("user_agent")
    @classmethod
    def _normalize_user_agent(cls, value: str) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_provider(self) -> ProviderConfigBase:
        if float(self.timeout_s) <= 0:
            raise ValueError("provider timeout_s must be > 0")
        if int(self.results_per_page) <= 0:
            raise ValueError("provider results_per_page must be > 0")
        if int(self.results_per_page) > 100:
            raise ValueError("provider results_per_page must be <= 100")
        return self


ProviderConfigT = TypeVar(
    "ProviderConfigT",
    bound=ProviderConfigBase,
    default=ProviderConfigBase,
)


class SearchProviderBase(ComponentBase[ProviderConfigT], ABC, Generic[ProviderConfigT]):
    __di_contract__ = True

    @abstractmethod
    async def asearch(
        self,
        *,
        query: str,
        page: int = 1,
        language: str = "",
        **kwargs: Any,
    ) -> SearchProviderResponse:
        raise NotImplementedError


__all__ = [
    "GoogleSafeSearchKey",
    "ProviderConfigBase",
    "RetrySettings",
    "SearchProviderBase",
]
