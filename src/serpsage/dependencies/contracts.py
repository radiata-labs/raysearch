from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeAlias, TypeVar, cast
from typing_extensions import override

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class InjectToken(Generic[T]):
    name: str

    def __post_init__(self) -> None:
        token = str(self.name or "").strip()
        if not token:
            raise ValueError("inject token name must be non-empty")
        object.__setattr__(self, "name", token)

    @property
    def debug_name(self) -> str:
        return self.name

    @override
    def __str__(self) -> str:
        return self.name


ServiceKey: TypeAlias = type[T] | InjectToken[T]


@dataclass(frozen=True, slots=True)
class InjectRequest(Generic[T]):
    key: ServiceKey[T] | None = None


def Inject(key: ServiceKey[T] | None = None) -> T:
    return cast("T", InjectRequest(key=key))


class BindingScope(StrEnum):
    SINGLETON = "singleton"
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class ServiceBinding:
    binding_id: str
    key: ServiceKey[Any]
    provider_cls: type[Any] | None = None
    instance: object = field(default_factory=lambda: _MISSING)
    alias_key: ServiceKey[Any] | None = None
    scope: BindingScope = BindingScope.SINGLETON
    overrides: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MultiBinding:
    binding_id: str
    key: ServiceKey[Any]
    order: int
    provider_cls: type[Any] | None = None
    instance: object = field(default_factory=lambda: _MISSING)
    alias_key: ServiceKey[Any] | None = None
    scope: BindingScope = BindingScope.SINGLETON
    overrides: dict[str, object] = field(default_factory=dict)


def is_service_key(value: object) -> bool:
    return isinstance(value, (InjectToken, type))


def assert_service_key(value: object) -> ServiceKey[Any]:
    if not is_service_key(value):
        raise TypeError("service key must be a type or InjectToken")
    return cast("ServiceKey[Any]", value)


def format_service_key(value: ServiceKey[Any]) -> str:
    if isinstance(value, InjectToken):
        return value.debug_name
    module = str(getattr(value, "__module__", "") or "")
    qualname = str(getattr(value, "__qualname__", "") or value.__name__)
    if module and module != "builtins":
        return f"{module}.{qualname}"
    return qualname


class _Missing:
    pass


_MISSING: object = _Missing()


__all__ = [
    "BindingScope",
    "Inject",
    "InjectRequest",
    "InjectToken",
    "MultiBinding",
    "ServiceBinding",
    "ServiceKey",
    "assert_service_key",
    "format_service_key",
    "is_service_key",
]
