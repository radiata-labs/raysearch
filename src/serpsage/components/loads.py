from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, TypeVar, cast

from pydantic import BaseModel, ValidationError, create_model

from serpsage.components.base import ComponentBase, ComponentConfigBase
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
    import httpx

    from serpsage.core.overrides import Overrides

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

ComponentClass: TypeAlias = type[ComponentBase[Any]]
ConfigModelT = TypeVar("ConfigModelT", bound=ComponentConfigBase)


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    family: str
    name: str
    cls: ComponentClass
    config_cls: type[ComponentConfigBase]
    config: ComponentConfigBase


class ComponentRegistry:
    def __init__(
        self,
        *,
        settings: AppSettings,
        components: tuple[ComponentClass, ...],
        overrides: Overrides | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.settings = settings
        self._components = components
        self._overrides = overrides
        self._env = dict(env or {})
        self._all_specs = self._build_specs()
        self._enabled_specs_by_family = self._group_enabled_specs()
        self._default_specs = self._resolve_default_specs()

    def all_specs(self) -> tuple[ComponentSpec, ...]:
        return self._all_specs

    def iter_enabled(self) -> tuple[ComponentSpec, ...]:
        return tuple(
            spec
            for family_specs in self._enabled_specs_by_family.values()
            for spec in family_specs
        )

    def enabled_specs(self, family: str) -> tuple[ComponentSpec, ...]:
        return self._enabled_specs_by_family.get(str(family), ())

    def default_spec(self, family: str) -> ComponentSpec:
        normalized_family = str(family).strip()
        spec = self._default_specs.get(normalized_family)
        if spec is None:
            raise RuntimeError(f"no enabled component configured for family `{family}`")
        return spec

    def family_name(self, family: str) -> str:
        return self.default_spec(family).name

    def resolve_default_config(
        self,
        family: str,
        *,
        expected_type: type[ConfigModelT],
    ) -> ConfigModelT:
        config = self.default_spec(family).config
        if not isinstance(config, expected_type):
            raise TypeError(
                f"default config for family `{family}` must be `{expected_type.__name__}`, "
                f"got `{type(config).__name__}`"
            )
        return config

    def http_override(self) -> httpx.AsyncClient | None:
        if self._overrides is None:
            return None
        return self._overrides.http

    def _build_specs(self) -> tuple[ComponentSpec, ...]:
        specs: list[ComponentSpec] = []
        for component in self._components:
            config_cls = cast(
                "type[ComponentConfigBase]",
                getattr(component, "Config", None),
            )
            family = str(config_cls.__setting_family__).strip()
            name = str(config_cls.__setting_name__).strip()
            family_settings = getattr(self.settings.components, family, None)
            raw_value = getattr(family_settings, name, None)
            raw_payload = _dump_model(raw_value)
            config = config_cls.from_raw(raw_payload, env=self._env)
            if family_settings is not None:
                setattr(family_settings, name, config)
            specs.append(
                ComponentSpec(
                    family=family,
                    name=name,
                    cls=component,
                    config_cls=config_cls,
                    config=config,
                )
            )
        return tuple(specs)

    def _group_enabled_specs(self) -> dict[str, tuple[ComponentSpec, ...]]:
        grouped: dict[str, list[ComponentSpec]] = {}
        for spec in self._all_specs:
            if not bool(spec.config.enabled):
                continue
            grouped.setdefault(spec.family, []).append(spec)
        return {family: tuple(specs) for family, specs in grouped.items()}

    def _resolve_default_specs(self) -> dict[str, ComponentSpec]:
        defaults: dict[str, ComponentSpec] = {}
        for family, specs in self._enabled_specs_by_family.items():
            configured_default = str(
                getattr(
                    getattr(self.settings.components, family, None),
                    "default",
                    "",
                )
                or ""
            ).strip()
            if configured_default:
                for spec in specs:
                    if spec.name == configured_default:
                        defaults[family] = spec
                        break
                else:
                    raise ValueError(
                        f"configured default `{configured_default}` for family "
                        f"`{family}` is not enabled"
                    )
                continue
            defaults[family] = specs[0]
        return defaults


def load_components() -> tuple[ComponentClass, ...]:
    items: list[ComponentClass] = []
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
            config_cls = getattr(value, "Config", None)
            if _is_setting_class(config_cls):
                family = config_cls.__setting_family__
                if family not in _FAMILY_SETTING_BASES:
                    raise TypeError(
                        f"{value.__name__} must expose a concrete Config class with "
                        "`__setting_family__` in `{_FAMILY_SETTING_BASES}`"
                    )
                items.append(value)
    return tuple(items)


def materialize_settings(
    *,
    settings: AppSettings | Mapping[str, Any],
    components: tuple[ComponentClass, ...],
) -> AppSettings:
    if isinstance(settings, AppSettings):
        raw_settings = settings.model_dump(mode="python")
    elif isinstance(settings, Mapping):
        raw_settings = dict(settings)
    else:
        raise TypeError(
            f"settings must be an AppSettings or mapping, got `{type(settings).__name__}`"
        )
    family_descriptors: dict[str, list[ComponentClass]] = {}
    for component in components:
        if (
            (config := getattr(component, "Config", None))
            and _is_setting_class(config)
            and config.__setting_family__ in _FAMILY_SETTING_BASES
        ):
            family_descriptors.setdefault(config.__setting_family__, []).append(
                component
            )

    component_fields: dict[str, Any] = {}
    for family_name in family_descriptors:
        field_definitions: dict[str, Any] = {}
        for component in family_descriptors.get(family_name, ()):
            config_class = cast(
                "type[ComponentConfigBase]", getattr(component, "Config", None)
            )
            try:
                default_value: Any = config_class()
            except ValidationError:
                default_value = ...
            field_definitions[config_class.__setting_name__] = (
                config_class,
                default_value,
            )

        base = _FAMILY_SETTING_BASES.get(family_name, SettingModel)
        family_model = cast(
            "type[SettingModel]",
            create_model(
                base.__name__,
                __base__=base,
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
    return app_model.model_validate(raw_settings)


def _dump_model(value: object) -> dict[str, Any] | None:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _is_setting_class(candidate: object) -> TypeGuard[type[ComponentConfigBase]]:
    return (
        inspect.isclass(candidate)
        and issubclass(candidate, ComponentConfigBase)
        and bool(str(getattr(candidate, "__setting_family__", "")).strip())
        and bool(str(getattr(candidate, "__setting_name__", "")).strip())
    )


__all__ = [
    "ComponentClass",
    "ComponentRegistry",
    "ComponentSpec",
    "load_components",
    "materialize_settings",
]
