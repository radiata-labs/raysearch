from __future__ import annotations

import contextlib
import importlib
import inspect
import os
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    get_origin,
    overload,
)

import httpx
from pydantic import ValidationError, create_model

from serpsage.components.base import (
    BUILTIN_COMPONENT_FAMILIES,
    ComponentBase,
    ComponentConfigBase,
    ComponentFamily,
    ComponentMeta,
    coerce_component_family,
)
from serpsage.dependencies import analyze_class
from serpsage.settings.models import (
    AppSettings,
    CacheSettings,
    ComponentSettings,
    CrawlSettings,
    ExtractSettings,
    HttpSettings,
    LlmSettings,
    ProviderSettings,
    RankSettings,
    RateLimitSettings,
    SettingModel,
    TelemetrySettings,
)

if TYPE_CHECKING:
    from types import ModuleType

    from serpsage.core.runtime import Overrides

ComponentClass: TypeAlias = type[ComponentBase[Any]]
_T = TypeVar("_T", bound=type[ComponentBase[Any]])
_ConfigT = TypeVar("_ConfigT", bound=ComponentConfigBase)
TypeMap = dict[object, Any]
_COMPONENT_META_ATTR = "__serpsage_component_meta__"

_FAMILY_SETTING_BASES: dict[str, type[SettingModel]] = {
    "cache": CacheSettings,
    "crawl": CrawlSettings,
    "extract": ExtractSettings,
    "http": HttpSettings,
    "llm": LlmSettings,
    "provider": ProviderSettings,
    "rank": RankSettings,
    "rate_limit": RateLimitSettings,
    "telemetry": TelemetrySettings,
}
_PREFERRED_DEFAULT_COMPONENTS = {
    "cache": "null",
    "crawl": "auto",
    "extract": "auto",
    "http": "httpx",
    "llm": "router",
    "provider": "searxng",
    "rank": "blend",
    "rate_limit": "basic",
    "telemetry": "async_emitter",
}


class _ComponentResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _ComponentDescriptor:
    family: ComponentFamily[Any]
    cls: ComponentClass
    config_cls: type[ComponentConfigBase]
    meta: ComponentMeta


@dataclass(frozen=True, slots=True)
class _ResolvedComponent:
    family: ComponentFamily[Any]
    component_name: str
    descriptor: _ComponentDescriptor
    config: ComponentConfigBase


class ComponentCatalog:
    def __init__(
        self,
        *,
        settings: AppSettings,
        descriptors: tuple[_ComponentDescriptor, ...],
        overrides: Overrides | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._descriptors = descriptors
        self._descriptors_by_family = _group_descriptors_by_family(descriptors)
        self._overrides = overrides
        self._env = dict(env or settings.runtime_env or os.environ)
        self._family_spec_cache: dict[str, tuple[_ResolvedComponent, ...]] = {}

    def resolve_default_spec(
        self,
        family: ComponentFamily[Any] | str,
    ) -> _ResolvedComponent:
        normalized = coerce_component_family(family)
        specs = self._family_specs(normalized)
        if not specs:
            raise _ComponentResolutionError(
                f"component family `{normalized.name}` has no enabled component"
            )
        preferred_name = _PREFERRED_DEFAULT_COMPONENTS.get(normalized.name)
        if preferred_name:
            for spec in specs:
                if spec.component_name == preferred_name:
                    return spec
        return specs[0]

    @overload
    def resolve_default_config(
        self,
        family: ComponentFamily[Any] | str,
        *,
        expected_type: type[_ConfigT],
    ) -> _ConfigT: ...

    @overload
    def resolve_default_config(
        self,
        family: ComponentFamily[Any] | str,
        *,
        expected_type: None = None,
    ) -> ComponentConfigBase: ...

    def resolve_default_config(
        self,
        family: ComponentFamily[Any] | str,
        *,
        expected_type: type[_ConfigT] | None = None,
    ) -> ComponentConfigBase | _ConfigT:
        config = self.resolve_default_spec(family).config
        if expected_type is None:
            return config
        if not isinstance(config, expected_type):
            raise TypeError(
                f"default config for `{coerce_component_family(family).name}` expected "
                f"`{expected_type.__name__}`, got `{type(config).__name__}`"
            )
        return config

    def family_name(self, family: ComponentFamily[Any] | str) -> str:
        return self.resolve_default_spec(family).component_name

    def component_families(self) -> tuple[ComponentFamily[Any], ...]:
        families: list[ComponentFamily[Any]] = [
            family
            for family in BUILTIN_COMPONENT_FAMILIES
            if self._family_specs(family)
        ]
        for family in _descriptor_families(self._descriptors):
            if family in BUILTIN_COMPONENT_FAMILIES:
                continue
            if self._family_specs(family):
                families.append(family)
        return tuple(families)

    def iter_enabled_specs(self) -> tuple[_ResolvedComponent, ...]:
        specs: list[_ResolvedComponent] = []
        for family in self.component_families():
            specs.extend(self._family_specs(family))
        return tuple(specs)

    def http_override(self) -> httpx.AsyncClient | None:
        overrides = self._overrides
        if overrides is None:
            return None
        client = overrides.http
        return client if isinstance(client, httpx.AsyncClient) else None

    def _family_specs(
        self,
        family: ComponentFamily[Any] | str,
    ) -> tuple[_ResolvedComponent, ...]:
        normalized = coerce_component_family(family)
        cached = self._family_spec_cache.get(normalized.name)
        if cached is not None:
            return cached
        family_settings = _family_settings(self._settings, normalized.name)
        specs: list[_ResolvedComponent] = []
        for descriptor in self._descriptors_by_family.get(normalized.name, ()):
            config = getattr(
                family_settings,
                descriptor.config_cls.__setting_name__,
                None,
            )
            if not isinstance(config, descriptor.config_cls):
                continue
            if not bool(config.enabled):
                continue
            specs.append(
                _ResolvedComponent(
                    family=normalized,
                    component_name=descriptor.config_cls.__setting_name__,
                    descriptor=descriptor,
                    config=descriptor.config_cls.from_raw(
                        config.model_dump(mode="python"),
                        env=self._env,
                    ),
                )
            )
        resolved = tuple(specs)
        self._family_spec_cache[normalized.name] = resolved
        return resolved


def load_component_descriptors() -> tuple[_ComponentDescriptor, ...]:
    items: dict[tuple[str, str], _ComponentDescriptor] = {}
    for module in _iter_component_modules():
        _register_module_components(items=items, module=module)
    return tuple(items.values())


def materialize_settings(
    *,
    settings: AppSettings | Mapping[str, Any],
    descriptors: tuple[_ComponentDescriptor, ...],
) -> AppSettings:
    if isinstance(settings, AppSettings):
        raw_settings = settings.model_dump(mode="python")
    elif isinstance(settings, Mapping):
        raw_settings = dict(settings)
    else:
        raise TypeError(
            f"settings must be an AppSettings or mapping, got `{type(settings).__name__}`"
        )
    app_model = _create_settings_model(descriptors=descriptors)
    return app_model.model_validate(raw_settings)


def register_component(*, meta: ComponentMeta) -> Callable[[_T], _T]:
    def decorator(cls: _T) -> _T:
        setattr(cls, _COMPONENT_META_ATTR, meta)
        return cls

    return decorator


def _create_settings_model(
    *,
    descriptors: tuple[_ComponentDescriptor, ...],
) -> type[AppSettings]:
    component_fields: dict[str, Any] = {}
    for family_name in _iter_component_family_names(descriptors):
        family_model = _build_family_settings_model(
            family_name=family_name,
            descriptors=_list_family_descriptors(descriptors, family_name),
        )
        component_fields[family_name] = (
            family_model,
            _default_model_instance(family_model),
        )
    components_model = cast(
        "type[ComponentSettings]",
        create_model(
            "ComponentSettings",
            __base__=ComponentSettings,
            **component_fields,
        ),
    )
    return cast(
        "type[AppSettings]",
        create_model(
            "AppSettings",
            __base__=AppSettings,
            components=(components_model, _default_model_instance(components_model)),
        ),
    )


def _iter_component_modules() -> tuple[ModuleType, ...]:
    modules: list[ModuleType] = []
    package = importlib.import_module("serpsage.components")
    for module_info in pkgutil.walk_packages(
        package.__path__,
        prefix=f"{package.__name__}.",
    ):
        if module_info.ispkg:
            continue
        with contextlib.suppress(ImportError):
            modules.append(importlib.import_module(module_info.name))
    return tuple(modules)


def _register_module_components(
    *,
    items: dict[tuple[str, str], _ComponentDescriptor],
    module: ModuleType,
) -> None:
    for value in module.__dict__.values():
        if not isinstance(value, type):
            continue
        if value.__module__ != module.__name__:
            continue
        if not issubclass(value, ComponentBase):
            continue
        meta = getattr(value, _COMPONENT_META_ATTR, None)
        if not isinstance(meta, ComponentMeta):
            continue
        _register_component_descriptor(items=items, meta=meta, cls=value)


def _register_component_descriptor(
    *,
    items: dict[tuple[str, str], _ComponentDescriptor],
    meta: ComponentMeta,
    cls: type[ComponentBase[Any]],
) -> None:
    config_cls = _resolve_registered_config_class(cls)
    family = coerce_component_family(config_cls.__setting_family__)
    key = (family.name, config_cls.__setting_name__)
    existing = items.get(key)
    if existing is not None and existing.cls is cls:
        return
    if existing is not None:
        raise ValueError(
            f"component `{family.name}:{config_cls.__setting_name__}` is already registered"
        )
    _validate_component_signature(cls=cls, config_cls=config_cls)
    items[key] = _ComponentDescriptor(
        family=family,
        cls=cls,
        config_cls=config_cls,
        meta=meta,
    )


def _resolve_registered_config_class(
    cls: type[Any],
) -> type[ComponentConfigBase]:
    config_cls = getattr(cls, "Config", None)
    if not _is_setting_class(config_cls):
        raise TypeError(
            f"{cls.__name__} must expose a concrete Config class with "
            "`__setting_family__` and `__setting_name__`"
        )
    return cast("type[ComponentConfigBase]", config_cls)


def _validate_component_signature(
    *,
    cls: ComponentClass,
    config_cls: type[ComponentConfigBase],
) -> None:
    _ = analyze_class(cls)
    config_type = _resolve_component_config_type(cls)
    if not isinstance(config_type, type):
        raise TypeError(
            f"{cls.__name__} must bind a concrete `ComponentBase[ConfigModel]` config type"
        )
    if not issubclass(config_cls, config_type):
        raise TypeError(
            f"{cls.__name__} config model `{config_cls.__name__}` is incompatible "
            f"with `{config_type.__name__}`"
        )


def _build_family_settings_model(
    *,
    family_name: str,
    descriptors: tuple[_ComponentDescriptor, ...],
) -> type[SettingModel]:
    field_definitions: dict[str, Any] = {}
    for descriptor in descriptors:
        default_value = _default_field_value(descriptor.config_cls)
        field_definitions[descriptor.config_cls.__setting_name__] = (
            descriptor.config_cls,
            default_value,
        )
    base = _FAMILY_SETTING_BASES.get(family_name, SettingModel)
    return cast(
        "type[SettingModel]",
        create_model(
            _family_model_name(family_name),
            __base__=base,
            **field_definitions,
        ),
    )


def _family_settings(settings: AppSettings, family_name: str) -> SettingModel:
    value = getattr(settings.components, family_name, None)
    if not isinstance(value, SettingModel):
        raise _ComponentResolutionError(
            f"settings family `{family_name}` is not available in components"
        )
    return value


def _iter_component_family_names(
    descriptors: tuple[_ComponentDescriptor, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for family_name in _FAMILY_SETTING_BASES:
        names.append(family_name)
        seen.add(family_name)
    for family in _descriptor_families(descriptors):
        if family.name in seen:
            continue
        seen.add(family.name)
        names.append(family.name)
    return tuple(names)


def _list_family_descriptors(
    descriptors: tuple[_ComponentDescriptor, ...],
    family: ComponentFamily[Any] | str,
) -> tuple[_ComponentDescriptor, ...]:
    family_name = coerce_component_family(family).name
    return tuple(
        descriptor
        for descriptor in descriptors
        if descriptor.family.name == family_name
    )


def _descriptor_families(
    descriptors: tuple[_ComponentDescriptor, ...],
) -> tuple[ComponentFamily[Any], ...]:
    unique: dict[str, ComponentFamily[Any]] = {}
    for descriptor in descriptors:
        unique[descriptor.family.name] = descriptor.family
    return tuple(sorted(unique.values(), key=lambda family: family.name))


def _group_descriptors_by_family(
    descriptors: tuple[_ComponentDescriptor, ...],
) -> dict[str, tuple[_ComponentDescriptor, ...]]:
    grouped: dict[str, list[_ComponentDescriptor]] = {}
    for descriptor in descriptors:
        grouped.setdefault(descriptor.family.name, []).append(descriptor)
    return {family_name: tuple(items) for family_name, items in grouped.items()}


def _family_model_name(family_name: str) -> str:
    parts = family_name.split("_")
    return f"Dynamic{''.join(part.capitalize() for part in parts)}Settings"


def _default_field_value(setting_class: type[ComponentConfigBase]) -> Any:
    try:
        return setting_class()
    except ValidationError:
        return ...


def _default_model_instance(model_cls: type[SettingModel]) -> SettingModel:
    try:
        return model_cls()
    except ValidationError:
        return model_cls.model_construct()


def _is_setting_class(candidate: object) -> bool:
    return (
        inspect.isclass(candidate)
        and issubclass(candidate, ComponentConfigBase)
        and bool(str(getattr(candidate, "__setting_family__", "")).strip())
        and bool(str(getattr(candidate, "__setting_name__", "")).strip())
    )


def _resolve_component_config_type(cls: type[Any]) -> type[ComponentConfigBase] | None:
    return _walk_component_config_type(cls, {})


def _walk_component_config_type(
    cls: type[Any],
    type_map: TypeMap,
) -> type[ComponentConfigBase] | None:
    for base in getattr(cls, "__orig_bases__", ()):
        origin = get_origin(base) or base
        params = getattr(origin, "__parameters__", ())
        args = tuple(_resolve_typevar(arg, type_map) for arg in get_args(base))
        local_map = dict(type_map)
        local_map.update(
            {
                param: arg
                for param, arg in zip(params, args, strict=False)
                if isinstance(param, TypeVar)
            }
        )
        if origin is ComponentBase:
            if not args:
                return None
            candidate = args[0]
            return (
                candidate
                if isinstance(candidate, type)
                and issubclass(candidate, ComponentConfigBase)
                else None
            )
        if isinstance(origin, type):
            resolved = _walk_component_config_type(origin, local_map)
            if resolved is not None:
                return resolved
    for base in cls.__bases__:
        if base is object:
            continue
        resolved = _walk_component_config_type(base, type_map)
        if resolved is not None:
            return resolved
    return None


def _resolve_typevar(value: Any, type_map: TypeMap) -> Any:
    current = value
    visited: set[int] = set()
    while isinstance(current, TypeVar):
        marker = id(current)
        if marker in visited:
            break
        visited.add(marker)
        mapped = type_map.get(current, current)
        if mapped is current:
            break
        current = mapped
    return current


__all__ = ["ComponentCatalog"]
