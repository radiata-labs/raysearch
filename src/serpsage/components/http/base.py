from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic
from typing_extensions import TypeVar

from serpsage.components.base import ComponentBase, ComponentConfigBase

if TYPE_CHECKING:
    import httpx


class HttpClientConfig(ComponentConfigBase):
    __setting_family__ = "http"
    __setting_name__ = "httpx"

    proxy: str | None = None
    trust_env: bool = False
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry_s: float = 5.0


HttpClientConfigT = TypeVar(
    "HttpClientConfigT",
    bound=HttpClientConfig,
    default=HttpClientConfig,
)


class HttpClientBase(ComponentBase[HttpClientConfigT], ABC, Generic[HttpClientConfigT]):
    @property
    @abstractmethod
    def client(self) -> httpx.AsyncClient:
        raise NotImplementedError


__all__ = ["HttpClientBase", "HttpClientConfig", "HttpClientConfigT"]
