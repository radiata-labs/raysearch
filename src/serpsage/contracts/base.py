from __future__ import annotations

from typing import Generic, TypeVar

from serpsage.contracts.protocols import Clock, Telemetry
from serpsage.settings.models import AppSettings

SettingsT = TypeVar("SettingsT")


class Component(Generic[SettingsT]):
    """Base class for all components.

    Enforces a consistent injection shape: settings + telemetry + clock.
    """

    def __init__(
        self,
        *,
        settings: AppSettings,
        telemetry: Telemetry,
        clock: Clock,
    ) -> None:
        self.settings = settings
        self.telemetry = telemetry
        self.clock = clock

    async def aclose(self) -> None:
        return


__all__ = ["Component", "SettingsT"]

