from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from pydantic import ConfigDict

from serpsage.core.workunit import WorkUnit
from serpsage.settings.models import SettingModel

ConfigT = TypeVar("ConfigT", bound="ComponentConfigBase")
WorkUnitT = TypeVar("WorkUnitT", bound=WorkUnit)


class ComponentConfigBase(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = False

    __setting_family__: str = ""
    __setting_name__: str = ""

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
CRAWL_FAMILY = define_component_family("crawl")
EXTRACT_FAMILY = define_component_family("extract")
RANK_FAMILY = define_component_family("rank")
LLM_FAMILY = define_component_family("llm")
CACHE_FAMILY = define_component_family("cache")
TELEMETRY_FAMILY = define_component_family("telemetry")
RATE_LIMIT_FAMILY = define_component_family("rate_limit")

BUILTIN_COMPONENT_FAMILIES = (
    HTTP_FAMILY,
    PROVIDER_FAMILY,
    CRAWL_FAMILY,
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
    config_optional: bool = False


class ComponentBase(WorkUnit, Generic[ConfigT]):
    meta: ClassVar[ComponentMeta]

    Config: type[ConfigT]

    def __init_subclass__(
        cls,
        config: type[ConfigT] | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init_subclass__()

        orig_bases: tuple[type, ...] = getattr(cls, "__orig_bases__", ())
        for orig_base in orig_bases:
            origin_class = get_origin(orig_base)
            if not (
                inspect.isclass(origin_class)
                and issubclass(origin_class, ComponentBase)
            ):
                continue
            try:
                config_t = cast("tuple[ConfigT]", get_args(orig_base))[0]
            except ValueError:  # pragma: no cover
                continue
            if (
                config is None
                and inspect.isclass(config_t)
                and issubclass(config_t, ComponentConfigBase)
            ):
                config = config_t
        if config is not None:
            cls.Config = _specialize_component_config(cls=cls, config=config)

    @property
    def config(self) -> ConfigT:
        default: Any = None
        config_class = getattr(self, "Config", None)
        if inspect.isclass(config_class) and issubclass(
            config_class, ComponentConfigBase
        ):
            value = getattr(
                getattr(
                    self.rt.settings.components, config_class.__setting_family__, None
                ),
                config_class.__setting_name__,
                default,
            )
            return cast("ConfigT", value)
        return cast("ConfigT", default)


def _specialize_component_config(
    *,
    cls: type[ComponentBase[Any]],
    config: type[ConfigT],
) -> type[ConfigT]:
    meta = getattr(cls, "meta", None)
    if not isinstance(meta, ComponentMeta):
        return config
    family_name = coerce_component_family(meta.family).name
    setting_name = str(meta.name or "").strip()
    if not family_name or not setting_name:
        raise TypeError(f"{cls.__name__} must declare a non-empty component meta")
    specialized = type(
        f"{cls.__name__}Config",
        (config,),
        {
            "__module__": cls.__module__,
            "__setting_family__": family_name,
            "__setting_name__": setting_name,
        },
    )
    return cast("type[ConfigT]", specialized)


__all__ = [
    "BUILTIN_COMPONENT_FAMILIES",
    "CACHE_FAMILY",
    "CRAWL_FAMILY",
    "ComponentBase",
    "ComponentConfigBase",
    "ComponentFamily",
    "ComponentMeta",
    "ConfigT",
    "EXTRACT_FAMILY",
    "HTTP_FAMILY",
    "LLM_FAMILY",
    "PROVIDER_FAMILY",
    "RANK_FAMILY",
    "RATE_LIMIT_FAMILY",
    "TELEMETRY_FAMILY",
    "coerce_component_family",
    "define_component_family",
]
