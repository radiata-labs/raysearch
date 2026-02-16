from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    import httpx


class HttpClientBase(WorkUnit, ABC):
    @property
    @abstractmethod
    def client(self) -> httpx.AsyncClient:
        raise NotImplementedError
