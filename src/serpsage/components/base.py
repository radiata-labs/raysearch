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
from serpsage.dependencies.contracts import InjectToken
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
_COMPONENT_COLLECTION_TOKENS: dict[str, InjectToken[tuple[object, ...]]] = {}


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


def family_collection_token(
    family: ComponentFamily[Any] | str,
) -> InjectToken[tuple[object, ...]]:
    normalized = coerce_component_family(family).name
    token = _COMPONENT_COLLECTION_TOKENS.get(normalized)
    if token is not None:
        return token
    created: InjectToken[tuple[object, ...]] = InjectToken(
        f"components.{normalized}.all"
    )
    _COMPONENT_COLLECTION_TOKENS[normalized] = created
    return created


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
    version: str
    summary: str


class ComponentBase(WorkUnit, Generic[ConfigT]):
    meta: ClassVar[ComponentMeta]
    Config: type[ConfigT]
    __di_contract__: ClassVar[bool] = False
    _component_config: ConfigT

    def __init_subclass__(
        cls,
        config: type[ConfigT] | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init_subclass__()
        resolved_config = config

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
                resolved_config is None
                and inspect.isclass(config_t)
                and issubclass(config_t, ComponentConfigBase)
            ):
                resolved_config = cast("type[ConfigT]", config_t)
        if resolved_config is not None:
            family_name = str(
                getattr(resolved_config, "__setting_family__", "")
            ).strip()
            setting_name = str(getattr(resolved_config, "__setting_name__", "")).strip()
            if not family_name or not setting_name:
                raise TypeError(
                    f"{cls.__name__} config `{resolved_config.__name__}` must declare "
                    "non-empty `__setting_family__` and `__setting_name__`"
                )
            cls.Config = resolved_config

    @property
    def config(self) -> ConfigT:
        try:
            return self._component_config
        except AttributeError as exc:  # pragma: no cover
            raise RuntimeError(
                f"{type(self).__name__} has no bound component config"
            ) from exc


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
    "family_collection_token",
]
