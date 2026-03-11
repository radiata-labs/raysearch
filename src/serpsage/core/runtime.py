from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field

from serpsage.models.base import MutableModel
from serpsage.settings.models import AppSettings

if TYPE_CHECKING:
    from serpsage.components.telemetry import TelemetryEmitterBase
    from serpsage.dependencies import ServiceProvider
    from serpsage.load import ComponentCatalog


class ClockBase(ABC):
    @abstractmethod
    def now_ms(self) -> int:
        raise NotImplementedError


class Runtime(MutableModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    settings: AppSettings
    clock: ClockBase
    telemetry: TelemetryEmitterBase[Any] | None = None
    components: ComponentCatalog | None = None
    services: ServiceProvider | None = None
    env: dict[str, str] = Field(default_factory=dict)


def _rebuild_runtime_model() -> None:
    from serpsage.dependencies.resolver import ServiceProvider
    from serpsage.load.components import ComponentCatalog

    Runtime.model_rebuild(
        _types_namespace={
            "ComponentCatalog": ComponentCatalog,
            "ServiceProvider": ServiceProvider,
        }
    )


_rebuild_runtime_model()

__all__ = ["ClockBase", "Runtime"]
