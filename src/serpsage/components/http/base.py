from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from serpsage.components.base import ComponentBase, ComponentConfigBase

if TYPE_CHECKING:
    import httpx


class HttpClientConfig(ComponentConfigBase):
    proxy: str | None = None
    trust_env: bool = False
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry_s: float = 5.0


class HttpClientBase(ComponentBase[HttpClientConfig], ABC):
    @property
    @abstractmethod
    def client(self) -> httpx.AsyncClient:
        raise NotImplementedError


__all__ = ["HttpClientBase", "HttpClientConfig"]
