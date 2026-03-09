from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
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

from serpsage.components.base import (
    BUILTIN_COMPONENT_FAMILIES,
    ComponentBase,
    ComponentConfigBase,
    ComponentFamily,
    ComponentMeta,
    coerce_component_family,
)
from serpsage.dependencies import analyze_class
from serpsage.settings.models import ComponentFamilySettings

if TYPE_CHECKING:
    from types import ModuleType

    from serpsage.core.runtime import Overrides
    from serpsage.settings.models import AppSettings

ComponentClass: TypeAlias = type[ComponentBase[Any]]
ComponentLoader: TypeAlias = Callable[["ComponentRegistry"], None]
_T = TypeVar("_T", bound=type[ComponentBase[Any]])
_ConfigT = TypeVar("_ConfigT", bound=ComponentConfigBase)
TypeMap = dict[object, Any]
_COMPONENT_META_ATTR = "__serpsage_component_meta__"

_BUILTIN_COMPONENT_MODULES = {
    ("cache", "memory"): "serpsage.components.cache.memory",
    ("cache", "mysql"): "serpsage.components.cache.mysql",
    ("cache", "null"): "serpsage.components.cache.null",
    ("cache", "redis"): "serpsage.components.cache.redis",
    ("cache", "sqlalchemy"): "serpsage.components.cache.sqlalchemy",
    ("cache", "sqlite"): "serpsage.components.cache.sqlite",
    ("extract", "auto"): "serpsage.components.extract.auto",
    ("extract", "html"): "serpsage.components.extract.html",
    ("extract", "pdf"): "serpsage.components.extract.pdf",
    ("fetch", "auto"): "serpsage.components.fetch.auto",
    ("fetch", "curl_cffi"): "serpsage.components.fetch.curl_cffi",
    ("fetch", "playwright"): "serpsage.components.fetch.playwright",
    ("http", "httpx"): "serpsage.components.http.client",
    ("llm", "dashscope"): "serpsage.components.llm.dashscope",
    ("llm", "gemini"): "serpsage.components.llm.gemini",
    ("llm", "openai"): "serpsage.components.llm.openai",
    ("llm", "router"): "serpsage.components.llm.router",
    ("provider", "google"): "serpsage.components.provider.google",
    ("provider", "searxng"): "serpsage.components.provider.searxng",
    ("rank", "blend"): "serpsage.components.rank.blend",
    ("rank", "bm25"): "serpsage.components.rank.bm25",
    ("rank", "cross_encoder"): "serpsage.components.rank.cross_encoder",
    ("rank", "heuristic"): "serpsage.components.rank.heuristic",
    ("rank", "tfidf"): "serpsage.components.rank.tfidf",
    ("rate_limit", "basic"): "serpsage.components.rate_limit.basic",
    ("telemetry", "async_emitter"): "serpsage.components.telemetry.emitter",
    ("telemetry", "jsonl_sink"): "serpsage.components.telemetry.sinks",
    ("telemetry", "null_emitter"): "serpsage.components.telemetry.emitter",
    ("telemetry", "null_sink"): "serpsage.components.telemetry.sinks",
    ("telemetry", "sqlite_metering_sink"): "serpsage.components.telemetry.sinks",
}
_OPTIONAL_BUILTIN_COMPONENTS = frozenset(
    {
        ("cache", "null"),
        ("extract", "html"),
        ("extract", "pdf"),
        ("fetch", "curl_cffi"),
        ("fetch", "playwright"),
        ("http", "httpx"),
        ("rank", "bm25"),
        ("rank", "cross_encoder"),
        ("rank", "heuristic"),
        ("rank", "tfidf"),
        ("rate_limit", "basic"),
        ("telemetry", "null_emitter"),
    }
)


class ComponentResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    family: ComponentFamily[Any]
    meta: ComponentMeta
    cls: ComponentClass


@dataclass(frozen=True, slots=True)
class ResolvedComponentSpec:
    family: ComponentFamily[Any]
    instance_id: str
    component_name: str
    descriptor: ComponentDescriptor
    config: ComponentConfigBase


class ComponentRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], ComponentDescriptor] = {}

    def register(self, meta: ComponentMeta, cls: _T) -> _T:
        family = coerce_component_family(meta.family)
        normalized_meta = replace(meta, family=family)
        self._validate_component_signature(meta=normalized_meta, cls=cls)
        key = (family.name, normalized_meta.name)
        existing = self._items.get(key)
        if existing is not None and existing.cls is cls:
            return cls
        if existing is not None:
            raise ValueError(
                f"component `{family.name}:{normalized_meta.name}` is already registered"
            )
        self._items[key] = ComponentDescriptor(
            family=family,
            meta=normalized_meta,
            cls=cls,
        )
        return cls

    def get(self, family: ComponentFamily[Any] | str, name: str) -> ComponentDescriptor:
        item = self.maybe_get(family, name)
        family_name = coerce_component_family(family).name
        if item is None:
            raise KeyError(f"component `{family_name}:{name}` is not registered")
        return item

    def maybe_get(
        self,
        family: ComponentFamily[Any] | str,
        name: str,
    ) -> ComponentDescriptor | None:
        family_name = coerce_component_family(family).name
        return self._items.get((family_name, str(name)))

    def list_family(
        self,
        family: ComponentFamily[Any] | str,
    ) -> list[ComponentDescriptor]:
        family_name = coerce_component_family(family).name
        return [item for key, item in self._items.items() if key[0] == family_name]

    def families(self) -> tuple[ComponentFamily[Any], ...]:
        unique: dict[str, ComponentFamily[Any]] = {}
        for item in self._items.values():
            unique[item.family.name] = item.family
        return tuple(sorted(unique.values(), key=lambda family: family.name))

    def _validate_component_signature(
        self,
        *,
        meta: ComponentMeta,
        cls: ComponentClass,
    ) -> None:
        _ = analyze_class(cls)
        config_type = _resolve_component_config_type(cls)
        family_name = cast("ComponentFamily[Any]", meta.family).name
        if not isinstance(config_type, type):
            raise TypeError(
                f"component `{family_name}:{meta.name}` must bind a concrete "
                "`ComponentBase[ConfigModel]` config type"
            )
        if not issubclass(meta.config_model, config_type):
            raise TypeError(
                f"component `{family_name}:{meta.name}` config model "
                f"`{meta.config_model.__name__}` is incompatible with component config "
                f"type `{config_type.__name__}`"
            )
        for contract in meta.contracts:
            if not isinstance(contract, type):
                raise TypeError(
                    f"component `{family_name}:{meta.name}` contracts must be types"
                )
            if not issubclass(cls, contract):
                raise TypeError(
                    f"component `{family_name}:{meta.name}` must implement contract "
                    f"`{contract.__name__}`"
                )


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
        descriptor = self._resolve_descriptor(
            family=normalized,
            component_name=str(instance_settings.component),
        )
        if (
            str(instance_id) not in family_settings.declared_instances
            and not descriptor.meta.config_optional
        ):
            raise ComponentResolutionError(
                f"component instance `{normalized.name}:{instance_id}` is not explicitly configured"
            )
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
                descriptor = self._registry.maybe_get(
                    family, instance_settings.component
                )
                if descriptor is None:
                    if instance_id in family_settings.declared_instances:
                        self._resolve_descriptor(
                            family=family,
                            component_name=str(instance_settings.component),
                        )
                    continue
                if (
                    instance_id not in family_settings.declared_instances
                    and not descriptor.meta.config_optional
                ):
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

    def _resolve_descriptor(
        self,
        *,
        family: ComponentFamily[Any],
        component_name: str,
    ) -> ComponentDescriptor:
        try:
            return self._registry.get(family, component_name)
        except KeyError as exc:
            raise ComponentResolutionError(
                f"component `{family.name}:{component_name}` is not available in this load session"
            ) from exc


ComponentContainer = ComponentCatalog


def load_component_registry(
    *,
    settings: AppSettings,
    component_loader: ComponentLoader | None = None,
) -> ComponentRegistry:
    registry = ComponentRegistry()
    imported_modules: set[str] = set()
    for module_path in _iter_builtin_module_paths(settings):
        if module_path in imported_modules:
            continue
        imported_modules.add(module_path)
        _register_module_components(
            registry=registry,
            module=importlib.import_module(module_path),
        )
    if component_loader is not None:
        component_loader(registry)
    return registry


def register_component(*, meta: ComponentMeta) -> Callable[[_T], _T]:
    def decorator(cls: _T) -> _T:
        setattr(cls, _COMPONENT_META_ATTR, meta)
        return cls

    return decorator


def _iter_builtin_module_paths(settings: AppSettings) -> tuple[str, ...]:
    selected: list[str] = []
    seen: set[str] = set()
    for family in BUILTIN_COMPONENT_FAMILIES:
        family_settings = getattr(settings, family.name, None)
        if not isinstance(family_settings, ComponentFamilySettings):
            continue
        for instance_id, instance_settings in family_settings.instances.items():
            key = (family.name, str(instance_settings.component))
            if (
                instance_id not in family_settings.declared_instances
                and key not in _OPTIONAL_BUILTIN_COMPONENTS
            ):
                continue
            module_path = _BUILTIN_COMPONENT_MODULES.get(key)
            if module_path is None or module_path in seen:
                continue
            seen.add(module_path)
            selected.append(module_path)
    return tuple(selected)


def _register_module_components(
    *,
    registry: ComponentRegistry,
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
        registry.register(meta, value)


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


__all__ = [
    "ComponentCatalog",
    "ComponentContainer",
    "ComponentDescriptor",
    "ComponentRegistry",
    "ComponentResolutionError",
    "ResolvedComponentSpec",
    "load_component_registry",
    "register_component",
]
