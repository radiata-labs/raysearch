from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeVar

from pydantic import Field, model_validator

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.provider import SearchProviderResponse

GoogleSafeSearchKey = Literal["off", "medium", "high"]
ProviderConfigT = TypeVar("ProviderConfigT", bound="ProviderConfigBase")


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

    @model_validator(mode="after")
    def _validate_provider(self) -> ProviderConfigBase:
        self.base_url = str(self.base_url or "").strip()
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError("provider base_url must start with http:// or https://")
        if float(self.timeout_s) <= 0:
            raise ValueError("provider timeout_s must be > 0")
        self.user_agent = str(self.user_agent or "").strip()
        if int(self.results_per_page) <= 0:
            raise ValueError("provider results_per_page must be > 0")
        if int(self.results_per_page) > 100:
            raise ValueError("provider results_per_page must be <= 100")
        return self


class SearchProviderBase(ComponentBase[ProviderConfigT], ABC, Generic[ProviderConfigT]):
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
