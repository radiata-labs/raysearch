from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast, get_args, get_origin

from serpsage.components.base import (
    ComponentBase,
    ComponentConfigBase,
    ComponentFamily,
    ComponentMeta,
    coerce_component_family,
)
from serpsage.dependencies import analyze_class

if TYPE_CHECKING:
    from collections.abc import Callable

ComponentClass: TypeAlias = type[ComponentBase[Any]]
_T = TypeVar("_T", bound=type[ComponentBase[Any]])


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    family: ComponentFamily[Any]
    meta: ComponentMeta
    cls: ComponentClass


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
        family_name = coerce_component_family(family).name
        item = self._items.get((family_name, str(name)))
        if item is None:
            raise KeyError(f"component `{family_name}:{name}` is not registered")
        return item

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
        if not isinstance(config_type, type):
            raise TypeError(
                f"component `{cast('ComponentFamily[Any]', meta.family).name}:{meta.name}` "
                "must bind a concrete `ComponentBase[ConfigModel]` config type"
            )
        if not issubclass(meta.config_model, config_type):
            raise TypeError(
                f"component `{cast('ComponentFamily[Any]', meta.family).name}:{meta.name}` config model "
                f"`{meta.config_model.__name__}` is incompatible with component config "
                f"type `{config_type.__name__}`"
            )
        for contract in meta.contracts:
            if not isinstance(contract, type):
                raise TypeError(
                    f"component `{cast('ComponentFamily[Any]', meta.family).name}:{meta.name}` "
                    "contracts must be types"
                )
            if not issubclass(cls, contract):
                raise TypeError(
                    f"component `{cast('ComponentFamily[Any]', meta.family).name}:{meta.name}` "
                    f"must implement contract `{contract.__name__}`"
                )


TypeMap = dict[TypeVar, Any]


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


_GLOBAL_COMPONENT_REGISTRY = ComponentRegistry()


def get_component_registry() -> ComponentRegistry:
    return _GLOBAL_COMPONENT_REGISTRY


def register_component(*, meta: ComponentMeta) -> Callable[[_T], _T]:
    def decorator(cls: _T) -> _T:
        return _GLOBAL_COMPONENT_REGISTRY.register(meta, cls)

    return decorator


__all__ = [
    "ComponentDescriptor",
    "ComponentRegistry",
    "get_component_registry",
    "register_component",
]
