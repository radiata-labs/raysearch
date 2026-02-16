from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SpanBase(ABC):
    @abstractmethod
    def add_event(self, name: str, **fields: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_attr(self, name: str, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def end(self) -> None:
        raise NotImplementedError


class TelemetryBase(ABC):
    @abstractmethod
    def start_span(self, name: str, **attrs: Any) -> SpanBase:
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        raise NotImplementedError


class ClockBase(ABC):
    @abstractmethod
    def now_ms(self) -> int:
        raise NotImplementedError


__all__ = ["ClockBase", "SpanBase", "TelemetryBase"]
