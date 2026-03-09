from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, TypeAlias, get_type_hints

from serpsage.components.base import (
    ComponentBase,
    ComponentConfigBase,
    ComponentMeta,
    is_dependency_request,
    unwrap_collection_annotation,
    unwrap_optional_annotation,
)

ComponentClass: TypeAlias = type[ComponentBase[Any]]


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    meta: ComponentMeta
    cls: ComponentClass


class ComponentRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], ComponentDescriptor] = {}

    def register(self, meta: ComponentMeta, cls: ComponentClass) -> ComponentClass:
        self._validate_component_signature(meta=meta, cls=cls)
        key = (meta.family, meta.name)
        existing = self._items.get(key)
        if existing is not None and existing.cls is cls:
            return cls
        if existing is not None:
            raise ValueError(
                f"component `{meta.family}:{meta.name}` is already registered"
            )
        self._items[key] = ComponentDescriptor(meta=meta, cls=cls)
        return cls

    def get(self, family: str, name: str) -> ComponentDescriptor:
        key = (str(family), str(name))
        item = self._items.get(key)
        if item is None:
            raise KeyError(f"component `{family}:{name}` is not registered")
        return item

    def list_family(self, family: str) -> list[ComponentDescriptor]:
        return [
            item for key, item in self._items.items() if key[0] == str(family or "")
        ]

    def _validate_component_signature(
        self,
        *,
        meta: ComponentMeta,
        cls: ComponentClass,
    ) -> None:
        signature = inspect.signature(cls.__init__)
        hints = get_type_hints(cls.__init__)
        config_param = signature.parameters.get("config")
        if config_param is None:
            raise TypeError(
                f"component `{meta.family}:{meta.name}` must declare a `config` parameter"
            )
        config_type = hints.get("config")
        if not isinstance(config_type, type) or not issubclass(
            config_type, ComponentConfigBase
        ):
            raise TypeError(
                f"component `{meta.family}:{meta.name}` config parameter must be a "
                "ComponentConfigBase subtype"
            )
        if not issubclass(meta.config_model, config_type):
            raise TypeError(
                f"component `{meta.family}:{meta.name}` config model "
                f"`{meta.config_model.__name__}` is incompatible with constructor "
                f"annotation `{config_type.__name__}`"
            )
        for name, parameter in signature.parameters.items():
            if name in {"self", "rt", "config", "bound_deps"}:
                continue
            default = parameter.default
            if not is_dependency_request(default):
                continue
            annotation = hints.get(name)
            if annotation is None:
                raise TypeError(
                    f"component `{meta.family}:{meta.name}` dependency `{name}` "
                    "must have a type annotation"
                )
            collection_origin, item_type = unwrap_collection_annotation(annotation)
            if collection_origin is not None:
                if item_type is None:
                    raise TypeError(
                        f"component `{meta.family}:{meta.name}` dependency `{name}` "
                        "must declare collection element type"
                    )
                continue
            resolved_type = unwrap_optional_annotation(annotation)
            if resolved_type is None:
                raise TypeError(
                    f"component `{meta.family}:{meta.name}` dependency `{name}` "
                    "must use a concrete class annotation"
                )
            _ = default


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
