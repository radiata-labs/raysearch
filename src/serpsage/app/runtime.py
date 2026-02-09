from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.contracts.protocols import Clock, Telemetry
    from serpsage.settings.models import AppSettings


@dataclass(frozen=True, slots=True)
class CoreRuntime:
    """Single injection carrier for all work units.

    Hard rule: work units must not read env or create telemetry/clock themselves.
    """

    settings: AppSettings
    telemetry: Telemetry
    clock: Clock


__all__ = ["CoreRuntime"]
