from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from raysearch.components.base import ComponentBase, ComponentConfigBase
from raysearch.models.components.metering import MeterRecord

MeteringSinkConfigT = TypeVar("MeteringSinkConfigT", bound=ComponentConfigBase)


class MeteringSinkBase(
    ComponentBase[MeteringSinkConfigT],
    ABC,
    Generic[MeteringSinkConfigT],
):
    """Base class for metering sink components."""

    @abstractmethod
    async def emit(self, *, record: MeterRecord) -> None:
        raise NotImplementedError


__all__ = ["MeteringSinkBase"]
