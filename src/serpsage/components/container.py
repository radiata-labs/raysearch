from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, overload

import httpx

from serpsage.components.base import (
    BUILTIN_COMPONENT_FAMILIES,
    ComponentConfigBase,
    ComponentFamily,
    coerce_component_family,
)
from serpsage.components.registry import ComponentDescriptor, ComponentRegistry
from serpsage.settings.models import ComponentFamilySettings

if TYPE_CHECKING:
    from serpsage.core.runtime import Overrides
    from serpsage.settings.models import AppSettings

TConfig = TypeVar("TConfig", bound=ComponentConfigBase)


class ComponentResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResolvedComponentSpec:
    family: ComponentFamily[Any]
    instance_id: str
    component_name: str
    descriptor: ComponentDescriptor
    config: ComponentConfigBase


class ComponentCatalog:
    def __init__(
        self,
        *,
        settings: AppSettings,
        registry: ComponentRegistry,
        overrides: Overrides | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._overrides = overrides
        self._env = dict(env or settings.runtime_env or os.environ)
        self._spec_cache: dict[tuple[str, str], ResolvedComponentSpec] = {}

    def resolve_default_spec(
        self,
        family: ComponentFamily[Any] | str,
    ) -> ResolvedComponentSpec:
        normalized = coerce_component_family(family)
        return self.resolve_spec(
            family=normalized,
            instance_id=self.family_settings(normalized).default,
        )

    @overload
    def resolve_default_config(
        self,
        family: ComponentFamily[Any] | str,
        *,
        expected_type: type[TConfig],
    ) -> TConfig: ...

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
        expected_type: type[TConfig] | None = None,
    ) -> ComponentConfigBase | TConfig:
        config = self.resolve_default_spec(family).config
        if expected_type is None:
            return config
        if not isinstance(config, expected_type):
            raise TypeError(
                f"default config for `{coerce_component_family(family).name}` expected "
                f"`{expected_type.__name__}`, got `{type(config).__name__}`"
            )
        return config

    def resolve_spec(
        self,
        *,
        family: ComponentFamily[Any] | str,
        instance_id: str,
    ) -> ResolvedComponentSpec:
        normalized = coerce_component_family(family)
        key = (normalized.name, str(instance_id))
        cached = self._spec_cache.get(key)
        if cached is not None:
            return cached
        family_settings = self.family_settings(normalized)
        instance_settings = family_settings.instances.get(str(instance_id))
        if instance_settings is None:
            raise ComponentResolutionError(
                f"component instance `{normalized.name}:{instance_id}` does not exist"
            )
        if not bool(instance_settings.enabled):
            raise ComponentResolutionError(
                f"component instance `{normalized.name}:{instance_id}` is disabled"
            )
        descriptor = self._registry.get(normalized, instance_settings.component)
        config = descriptor.meta.config_model.from_raw(
            dict(instance_settings.config or {}),
            env=self._env,
        )
        spec = ResolvedComponentSpec(
            family=normalized,
            instance_id=str(instance_id),
            component_name=str(instance_settings.component),
            descriptor=descriptor,
            config=config,
        )
        self._spec_cache[key] = spec
        return spec

    def family_settings(
        self,
        family: ComponentFamily[Any] | str,
    ) -> ComponentFamilySettings:
        family_name = coerce_component_family(family).name
        value = getattr(self._settings, family_name, None)
        if not isinstance(value, ComponentFamilySettings):
            raise ComponentResolutionError(
                f"settings family `{family_name}` is not configured as a component family"
            )
        return value

    def family_name(self, family: ComponentFamily[Any] | str) -> str:
        return self.resolve_default_spec(family).component_name

    def component_families(self) -> tuple[ComponentFamily[Any], ...]:
        return tuple(
            family
            for family in BUILTIN_COMPONENT_FAMILIES
            if isinstance(
                getattr(self._settings, family.name, None), ComponentFamilySettings
            )
        )

    def iter_enabled_specs(self) -> tuple[ResolvedComponentSpec, ...]:
        specs: list[ResolvedComponentSpec] = []
        for family in self.component_families():
            family_settings = self.family_settings(family)
            for instance_id, instance_settings in family_settings.instances.items():
                if not bool(instance_settings.enabled):
                    continue
                specs.append(
                    self.resolve_spec(family=family, instance_id=str(instance_id))
                )
        return tuple(specs)

    def http_override(self) -> httpx.AsyncClient | None:
        overrides = self._overrides
        if overrides is None:
            return None
        client = overrides.http
        return client if isinstance(client, httpx.AsyncClient) else None


ComponentContainer = ComponentCatalog


__all__ = [
    "ComponentCatalog",
    "ComponentContainer",
    "ComponentResolutionError",
    "ResolvedComponentSpec",
]
