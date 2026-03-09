from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TypeAlias, cast

from serpsage.components.base import (
    ComponentBase,
    ComponentFamily,
    ComponentMeta,
    coerce_component_family,
)
from serpsage.dependencies import analyze_constructor

ComponentClass: TypeAlias = type[ComponentBase[Any]]


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    family: ComponentFamily[Any]
    meta: ComponentMeta
    cls: ComponentClass


class ComponentRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], ComponentDescriptor] = {}

    def register(self, meta: ComponentMeta, cls: ComponentClass) -> ComponentClass:
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
        plan = analyze_constructor(cls)
        config_param = next(
            (parameter for parameter in plan.parameters if parameter.name == "config"),
            None,
        )
        if config_param is None or not isinstance(config_param.annotation, type):
            raise TypeError(
                f"component `{cast('ComponentFamily[Any]', meta.family).name}:{meta.name}` "
                "must declare a `config` parameter"
            )
        if not issubclass(meta.config_model, config_param.annotation):
            raise TypeError(
                f"component `{cast('ComponentFamily[Any]', meta.family).name}:{meta.name}` config model "
                f"`{meta.config_model.__name__}` is incompatible with constructor "
                f"annotation `{config_param.annotation.__name__}`"
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


_GLOBAL_COMPONENT_REGISTRY = ComponentRegistry()


def get_component_registry() -> ComponentRegistry:
    return _GLOBAL_COMPONENT_REGISTRY


def register_component(*, meta: ComponentMeta) -> Any:
    def decorator(cls: ComponentClass) -> ComponentClass:
        return _GLOBAL_COMPONENT_REGISTRY.register(meta, cls)

    return decorator


__all__ = [
    "ComponentDescriptor",
    "ComponentRegistry",
    "get_component_registry",
    "register_component",
]
