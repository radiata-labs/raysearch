from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from serpsage.contracts.lifecycle import ClockBase, SpanBase, TelemetryBase
    from serpsage.core.runtime import CoreRuntime
    from serpsage.settings.models import AppSettings


class WorkUnit:
    def __init__(self, *, rt: CoreRuntime) -> None:
        self.rt = rt

    @property
    def settings(self) -> AppSettings:
        return self.rt.settings

    @property
    def telemetry(self) -> TelemetryBase:
        return self.rt.telemetry

    @property
    def clock(self) -> ClockBase:
        return self.rt.clock

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[SpanBase]:
        sp = self.telemetry.start_span(name, **attrs)
        try:
            yield sp
        finally:
            sp.end()

    async def ainit(self) -> None:
        return

    async def aclose(self) -> None:
        return

    async def __aenter__(self) -> Self:
        await self.ainit()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


__all__ = ["WorkUnit"]
