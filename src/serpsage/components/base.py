from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from pydantic import ConfigDict

from serpsage.core.workunit import WorkUnit
from serpsage.settings.models import Model

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime

ConfigT = TypeVar("ConfigT", bound="ComponentConfigBase")
WorkUnitT = TypeVar("WorkUnitT", bound=WorkUnit)


class ComponentConfigBase(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    def inject_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> dict[str, Any]:
        _ = env
        return dict(raw)

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any] | None = None,
        *,
        env: dict[str, str],
    ) -> ComponentConfigBase:
        payload = cls.inject_env(dict(raw or {}), env=env)
        return cls.model_validate(payload)


@dataclass(frozen=True, slots=True)
class ComponentFamily(Generic[WorkUnitT]):
    name: str

    def __post_init__(self) -> None:
        if not str(self.name or "").strip():
            raise ValueError("component family name must be non-empty")


_COMPONENT_FAMILIES: dict[str, ComponentFamily[Any]] = {}


def define_component_family(name: str) -> ComponentFamily[Any]:
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("component family name must be non-empty")
    family = _COMPONENT_FAMILIES.get(normalized)
    if family is not None:
        return family
    created = ComponentFamily[Any](name=normalized)
    _COMPONENT_FAMILIES[normalized] = created
    return created


def coerce_component_family(
    family: ComponentFamily[Any] | str,
) -> ComponentFamily[Any]:
    return (
        family
        if isinstance(family, ComponentFamily)
        else define_component_family(family)
    )


HTTP_FAMILY = define_component_family("http")
PROVIDER_FAMILY = define_component_family("provider")
FETCH_FAMILY = define_component_family("fetch")
EXTRACT_FAMILY = define_component_family("extract")
RANK_FAMILY = define_component_family("rank")
LLM_FAMILY = define_component_family("llm")
CACHE_FAMILY = define_component_family("cache")
TELEMETRY_FAMILY = define_component_family("telemetry")
RATE_LIMIT_FAMILY = define_component_family("rate_limit")

BUILTIN_COMPONENT_FAMILIES = (
    HTTP_FAMILY,
    PROVIDER_FAMILY,
    FETCH_FAMILY,
    EXTRACT_FAMILY,
    RANK_FAMILY,
    LLM_FAMILY,
    CACHE_FAMILY,
    TELEMETRY_FAMILY,
    RATE_LIMIT_FAMILY,
)


@dataclass(frozen=True, slots=True)
class ComponentMeta:
    family: ComponentFamily[Any] | str
    name: str
    version: str
    summary: str
    provides: tuple[str, ...] = ()
    contracts: tuple[type[Any], ...] = ()
    config_model: type[ComponentConfigBase] = ComponentConfigBase
    priority: int = 100
    enabled_by_default: bool = True


class ComponentBase(WorkUnit, Generic[ConfigT]):
    meta: ClassVar[ComponentMeta]

    def __init__(
        self,
        *,
        rt: Runtime | object,
        config: ConfigT,
    ) -> None:
        super().__init__(rt=cast("Runtime", rt))
        self.config = config


__all__ = [
    "BUILTIN_COMPONENT_FAMILIES",
    "CACHE_FAMILY",
    "ComponentBase",
    "ComponentConfigBase",
    "ComponentFamily",
    "ComponentMeta",
    "ConfigT",
    "EXTRACT_FAMILY",
    "FETCH_FAMILY",
    "HTTP_FAMILY",
    "LLM_FAMILY",
    "PROVIDER_FAMILY",
    "RANK_FAMILY",
    "RATE_LIMIT_FAMILY",
    "TELEMETRY_FAMILY",
    "coerce_component_family",
    "define_component_family",
]
