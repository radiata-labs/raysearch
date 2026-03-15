from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Generic, TypeVar

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.metering import (
    MeterName,
    MeterRecord,
    MeterUnit,
    TokenUsage,
)

MeteringSinkConfigT = TypeVar("MeteringSinkConfigT", bound=ComponentConfigBase)
MeteringEmitterConfigT = TypeVar("MeteringEmitterConfigT", bound=ComponentConfigBase)


class MeteringSinkBase(
    ComponentBase[MeteringSinkConfigT],
    ABC,
    Generic[MeteringSinkConfigT],
):
    @abstractmethod
    async def emit(self, *, record: MeterRecord) -> None:
        raise NotImplementedError


class MeteringEmitterBase(
    ComponentBase[MeteringEmitterConfigT],
    ABC,
    Generic[MeteringEmitterConfigT],
):
    @abstractmethod
    async def emit_record(self, *, record: MeterRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def push_request_context(self, *, request_id: str) -> object:
        raise NotImplementedError

    @abstractmethod
    def pop_request_context(self, token: object) -> None:
        raise NotImplementedError

    async def record(
        self,
        *,
        name: MeterName,
        unit: MeterUnit,
        request_id: str = "",
        key: str = "",
        provider: str = "",
        model: str = "",
        tokens: TokenUsage | dict[str, int] | None = None,
    ) -> None:
        with suppress(Exception):
            await self.emit_record(
                record=MeterRecord(
                    id=uuid.uuid4().hex,
                    ts_ms=int(self.clock.now_ms()),
                    name=name,
                    unit=unit,
                    request_id=request_id,
                    key=key,
                    provider=provider,
                    model=model,
                    tokens=(
                        TokenUsage.model_validate(tokens)
                        if tokens is not None
                        else None
                    ),
                )
            )


__all__ = ["MeteringEmitterBase", "MeteringSinkBase"]
