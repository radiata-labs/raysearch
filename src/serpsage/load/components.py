from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, TypeVar, cast, overload

from pydantic import ValidationError, create_model

from serpsage.components.base import (
    BUILTIN_COMPONENT_FAMILIES,
    ComponentBase,
    ComponentConfigBase,
    ComponentFamily,
    ComponentMeta,
    coerce_component_family,
)
from serpsage.dependencies.contracts import InjectToken
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
    from serpsage.core.runtime import Runtime

ComponentClass: TypeAlias = type[ComponentBase[Any]]
_ConfigT = TypeVar("_ConfigT", bound=ComponentConfigBase)
_ComponentT = TypeVar("_ComponentT")

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


class ComponentResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    family: ComponentFamily[Any]
    cls: ComponentClass
    config_cls: type[ComponentConfigBase]
    meta: ComponentMeta


@dataclass(frozen=True, slots=True)
class ResolvedComponent:
    family: ComponentFamily[Any]
    component_name: str
    descriptor: ComponentDescriptor
    config: ComponentConfigBase


class ComponentRegistry:
    def __init__(self, *, descriptors: tuple[ComponentDescriptor, ...]) -> None:
        self._descriptors = descriptors
        grouped: dict[str, list[ComponentDescriptor]] = {}
        dynamic_families: dict[str, ComponentFamily[Any]] = {}
        for descriptor in descriptors:
            grouped.setdefault(descriptor.family.name, []).append(descriptor)
            dynamic_families[descriptor.family.name] = descriptor.family
        self._descriptors_by_family = {
            family_name: tuple(items) for family_name, items in grouped.items()
        }
        ordered_families: list[ComponentFamily[Any]] = list(BUILTIN_COMPONENT_FAMILIES)
        for family in sorted(dynamic_families.values(), key=lambda item: item.name):
            if family in ordered_families:
                continue
            ordered_families.append(family)
        self._ordered_families = tuple(ordered_families)
        self._settings_model: type[AppSettings] | None = None

    @classmethod
    def discover(cls) -> ComponentRegistry:
        return cls(descriptors=_load_component_descriptors())

    def component_families(self) -> tuple[ComponentFamily[Any], ...]:
        return self._ordered_families

    def descriptors_for_family(
        self,
        family: ComponentFamily[Any] | str,
    ) -> tuple[ComponentDescriptor, ...]:
        normalized = coerce_component_family(family)
        return self._descriptors_by_family.get(normalized.name, ())

    def materialize_settings(
        self,
        settings: AppSettings | Mapping[str, Any],
    ) -> AppSettings:
        if isinstance(settings, AppSettings):
            raw_settings = settings.model_dump(mode="python")
        elif isinstance(settings, Mapping):
            raw_settings = dict(settings)
        else:
            raise TypeError(
                f"settings must be an AppSettings or mapping, got `{type(settings).__name__}`"
            )
        return self._create_settings_model().model_validate(raw_settings)

    def _create_settings_model(self) -> type[AppSettings]:
        cached = self._settings_model
        if cached is not None:
            return cached
        family_names: list[str] = list(_FAMILY_SETTING_BASES)
        seen = set(family_names)
        for family in self._ordered_families:
            if family.name in seen:
                continue
            family_names.append(family.name)
            seen.add(family.name)

        component_fields: dict[str, Any] = {}
        for family_name in family_names:
            field_definitions: dict[str, Any] = {}
            for descriptor in self._descriptors_by_family.get(family_name, ()):
                try:
                    default_value: Any = descriptor.config_cls()
                except ValidationError:
                    default_value = ...
                field_definitions[descriptor.config_cls.__setting_name__] = (
                    descriptor.config_cls,
                    default_value,
                )
            model_name = "Dynamic" + "".join(
                part.capitalize() for part in family_name.split("_")
            )
            family_model = cast(
                "type[SettingModel]",
                create_model(
                    f"{model_name}Settings",
                    __base__=_FAMILY_SETTING_BASES.get(family_name, SettingModel),
                    **field_definitions,
                ),
            )
            try:
                family_default = family_model()
            except ValidationError:
                family_default = family_model.model_construct()
            component_fields[family_name] = (family_model, family_default)

        components_model = cast(
            "type[ComponentSettings]",
            create_model(
                "ComponentSettings",
                __base__=ComponentSettings,
                **component_fields,
            ),
        )
        try:
            components_default = components_model()
        except ValidationError:
            components_default = components_model.model_construct()
        app_model = cast(
            "type[AppSettings]",
            create_model(
                "AppSettings",
                __base__=AppSettings,
                components=(components_model, components_default),
            ),
        )
        self._settings_model = app_model
        return app_model


class ComponentCatalog:
    def __init__(
        self,
        *,
        settings: AppSettings,
        registry: ComponentRegistry,
        env: dict[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._env = dict(env or settings.runtime_env or os.environ)
        self._runtime: Runtime | None = None
        self._family_spec_cache: dict[str, tuple[ResolvedComponent, ...]] = {}
        self._instance_keys: dict[tuple[str, str], InjectToken[object]] = {}

    def attach_runtime(self, runtime: Runtime) -> None:
        self._runtime = runtime

    def resolve_default_spec(
        self,
        family: ComponentFamily[Any] | str,
    ) -> ResolvedComponent:
        normalized = coerce_component_family(family)
        specs = self._family_specs(normalized)
        if not specs:
            raise ComponentResolutionError(
                f"component family `{normalized.name}` has no enabled component"
            )
        family_settings = self._family_settings(normalized.name)
        configured_name_raw = getattr(family_settings, "default", "")
        configured_name = (
            "" if configured_name_raw is None else str(configured_name_raw).strip()
        )
        if configured_name:
            for spec in specs:
                if spec.component_name == configured_name:
                    return spec
            raise ComponentResolutionError(
                f"component family `{normalized.name}` configured default "
                f"`{configured_name}` is not enabled or not registered"
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
        return tuple(
            family
            for family in self._registry.component_families()
            if self._family_specs(family)
        )

    def iter_enabled_specs(
        self,
        family: ComponentFamily[Any] | str | None = None,
    ) -> tuple[ResolvedComponent, ...]:
        if family is not None:
            return self._family_specs(family)
        specs: list[ResolvedComponent] = []
        for item in self.component_families():
            specs.extend(self._family_specs(item))
        return tuple(specs)

    def instance_key(
        self,
        family: ComponentFamily[Any] | str,
        component_name: str,
    ) -> InjectToken[object]:
        normalized = coerce_component_family(family)
        key = (normalized.name, str(component_name).strip())
        token = self._instance_keys.get(key)
        if token is not None:
            return token
        created: InjectToken[object] = InjectToken(
            f"component.{normalized.name}.{key[1]}"
        )
        self._instance_keys[key] = created
        return created

    @overload
    def require_component(
        self,
        family: ComponentFamily[Any] | str,
        component_name: str,
        *,
        expected_type: type[_ComponentT],
    ) -> _ComponentT: ...

    @overload
    def require_component(
        self,
        family: ComponentFamily[Any] | str,
        component_name: str,
        *,
        expected_type: None = None,
    ) -> object: ...

    def require_component(
        self,
        family: ComponentFamily[Any] | str,
        component_name: str,
        *,
        expected_type: type[_ComponentT] | None = None,
    ) -> object | _ComponentT:
        self._find_spec(family, component_name)
        services = self._require_runtime().services
        if services is None:
            raise RuntimeError("service provider is not attached to the runtime")
        value = services.require(self.instance_key(family, component_name))
        return _cast_component_value(
            value=value,
            family=coerce_component_family(family).name,
            component_name=component_name,
            expected_type=expected_type,
        )

    @overload
    def require_component_optional(
        self,
        family: ComponentFamily[Any] | str,
        component_name: str,
        *,
        expected_type: type[_ComponentT],
    ) -> _ComponentT | None: ...

    @overload
    def require_component_optional(
        self,
        family: ComponentFamily[Any] | str,
        component_name: str,
        *,
        expected_type: None = None,
    ) -> object | None: ...

    def require_component_optional(
        self,
        family: ComponentFamily[Any] | str,
        component_name: str,
        *,
        expected_type: type[_ComponentT] | None = None,
    ) -> object | _ComponentT | None:
        try:
            self._find_spec(family, component_name)
        except ComponentResolutionError:
            return None
        services = self._require_runtime().services
        if services is None:
            raise RuntimeError("service provider is not attached to the runtime")
        value = services.require_optional(self.instance_key(family, component_name))
        if value is None:
            return None
        return _cast_component_value(
            value=value,
            family=coerce_component_family(family).name,
            component_name=component_name,
            expected_type=expected_type,
        )

    def require_default_component(
        self,
        family: ComponentFamily[Any] | str,
    ) -> object:
        spec = self.resolve_default_spec(family)
        return self.require_component(spec.family, spec.component_name)

    def _find_spec(
        self,
        family: ComponentFamily[Any] | str,
        component_name: str,
    ) -> ResolvedComponent:
        normalized = coerce_component_family(family)
        target = str(component_name).strip()
        for spec in self._family_specs(normalized):
            if spec.component_name == target:
                return spec
        raise ComponentResolutionError(
            f"component `{normalized.name}:{target}` is not enabled or not registered"
        )

    def _family_specs(
        self,
        family: ComponentFamily[Any] | str,
    ) -> tuple[ResolvedComponent, ...]:
        normalized = coerce_component_family(family)
        cached = self._family_spec_cache.get(normalized.name)
        if cached is not None:
            return cached
        family_settings = self._family_settings(normalized.name)
        specs: list[ResolvedComponent] = []
        for descriptor in self._registry.descriptors_for_family(normalized):
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
                ResolvedComponent(
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

    def _family_settings(self, family_name: str) -> SettingModel:
        value = getattr(self._settings.components, family_name, None)
        if not isinstance(value, SettingModel):
            raise ComponentResolutionError(
                f"settings family `{family_name}` is not available in components"
            )
        return value

    def _require_runtime(self) -> Runtime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("component catalog runtime is not attached")
        return runtime


def _load_component_descriptors() -> tuple[ComponentDescriptor, ...]:
    items: dict[tuple[str, str], ComponentDescriptor] = {}
    package = importlib.import_module("serpsage.components")
    module_infos = sorted(
        pkgutil.walk_packages(
            package.__path__,
            prefix=f"{package.__name__}.",
        ),
        key=lambda item: item.name,
    )
    for module_info in module_infos:
        if module_info.ispkg:
            continue
        try:
            module = importlib.import_module(module_info.name)
        except Exception as exc:
            raise RuntimeError(
                f"failed to import component module `{module_info.name}`"
            ) from exc
        for value in module.__dict__.values():
            if not (
                isinstance(value, type)
                and value.__module__ == module.__name__
                and issubclass(value, ComponentBase)
                and ABC not in value.__bases__
                and not inspect.isabstract(value)
            ):
                continue
            meta = value.__dict__.get("meta")
            if not isinstance(meta, ComponentMeta):
                continue
            config_cls = getattr(value, "Config", None)
            if not _is_setting_class(config_cls):
                raise TypeError(
                    f"{value.__name__} must expose a concrete Config class with "
                    "`__setting_family__` and `__setting_name__`"
                )
            family = coerce_component_family(config_cls.__setting_family__)
            key = (family.name, config_cls.__setting_name__)
            existing = items.get(key)
            if existing is not None and existing.cls is value:
                continue
            if existing is not None:
                raise ValueError(
                    f"component `{family.name}:{config_cls.__setting_name__}` is already registered"
                )
            items[key] = ComponentDescriptor(
                family=family,
                cls=value,
                config_cls=config_cls,
                meta=meta,
            )
    return tuple(items[key] for key in sorted(items))


def _cast_component_value(
    *,
    value: object,
    family: str,
    component_name: str,
    expected_type: type[_ComponentT] | None,
) -> object | _ComponentT:
    if expected_type is None:
        return value
    if not isinstance(value, expected_type):
        raise TypeError(
            f"component `{family}:{component_name}` expected "
            f"`{expected_type.__name__}`, got `{type(value).__name__}`"
        )
    return value


def _is_setting_class(candidate: object) -> TypeGuard[type[ComponentConfigBase]]:
    return (
        inspect.isclass(candidate)
        and issubclass(candidate, ComponentConfigBase)
        and bool(str(getattr(candidate, "__setting_family__", "")).strip())
        and bool(str(getattr(candidate, "__setting_name__", "")).strip())
    )


__all__ = [
    "ComponentCatalog",
    "ComponentDescriptor",
    "ComponentRegistry",
    "ComponentResolutionError",
    "ResolvedComponent",
]
