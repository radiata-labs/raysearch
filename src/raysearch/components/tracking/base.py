from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from raysearch.components.base import ComponentBase, ComponentConfigBase
from raysearch.models.components.tracking import EventEnvelope

TrackingSinkConfigT = TypeVar("TrackingSinkConfigT", bound=ComponentConfigBase)


class TrackingSinkBase(
    ComponentBase[TrackingSinkConfigT],
    ABC,
    Generic[TrackingSinkConfigT],
):
    """Base class for tracking sink components."""

    @abstractmethod
    async def emit(self, *, event: EventEnvelope) -> None:
        raise NotImplementedError


__all__ = ["TrackingSinkBase"]
