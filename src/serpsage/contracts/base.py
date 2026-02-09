from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from serpsage.app.runtime import CoreRuntime
    from serpsage.contracts.protocols import Clock, Span, Telemetry
    from serpsage.settings.models import AppSettings


class WorkUnit:
    """Base class for all "working" classes.

    Hard injection shape: everything gets settings/telemetry/clock via CoreRuntime.
    """

    def __init__(self, *, rt: CoreRuntime) -> None:
        self.rt = rt

    @property
    def settings(self) -> AppSettings:
        return self.rt.settings

    @property
    def telemetry(self) -> Telemetry:
        return self.rt.telemetry

    @property
    def clock(self) -> Clock:
        return self.rt.clock

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        sp = self.telemetry.start_span(name, **attrs)
        try:
            yield sp
        finally:
            sp.end()

    async def aclose(self) -> None:
        return


__all__ = ["WorkUnit"]
