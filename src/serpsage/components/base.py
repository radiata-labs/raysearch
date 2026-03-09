from __future__ import annotations

from dataclasses import dataclass, field
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import ConfigDict

from serpsage.core.workunit import WorkUnit
from serpsage.settings.models import Model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from serpsage.core.runtime import Runtime

ConfigT = TypeVar("ConfigT", bound="ComponentConfigBase")


class ComponentConfigBase(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

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
class DependencySpec:
    name: str
    contract: str
    family: str | None = None
    required: bool = True
    multiple: bool = False


@dataclass(frozen=True, slots=True)
class ComponentMeta:
    family: str
    name: str
    version: str
    summary: str
    provides: tuple[str, ...]
    depends: tuple[DependencySpec, ...] = field(default_factory=tuple)
    config_model: type[ComponentConfigBase] = ComponentConfigBase
    priority: int = 100
    enabled_by_default: bool = True


@dataclass(frozen=True, slots=True)
class DependencyRequest:
    instance: str | None = None
    optional: bool = False
    use_cache: bool = True


@dataclass(frozen=True, slots=True)
class InjectedParams:
    kwargs: dict[str, Any]
    bound_deps: tuple[WorkUnit, ...] = ()


def Depends(
    *,
    instance: str | None = None,
    optional: bool = False,
    use_cache: bool = True,
) -> Any:
    return DependencyRequest(
        instance=instance,
        optional=bool(optional),
        use_cache=bool(use_cache),
    )


def is_dependency_request(value: object) -> bool:
    return isinstance(value, DependencyRequest)


def unwrap_collection_annotation(
    annotation: Any,
) -> tuple[type[Any] | None, type[Any] | None]:
    origin = get_origin(annotation)
    if origin not in {tuple, list}:
        return None, None
    args = get_args(annotation)
    if origin is tuple:
        if len(args) != 2 or args[1] is not Ellipsis:
            return tuple, None
        item_type = args[0]
    else:
        if len(args) != 1:
            return list, None
        item_type = args[0]
    return cast("type[Any]", origin), item_type if isinstance(item_type, type) else None


def unwrap_optional_annotation(annotation: Any) -> type[Any] | None:
    origin = get_origin(annotation)
    if origin not in {UnionType, Union}:
        return annotation if isinstance(annotation, type) else None
    args = [item for item in get_args(annotation) if item is not type(None)]
    if len(args) != 1:
        return None
    item_type = args[0]
    return item_type if isinstance(item_type, type) else None


class ComponentBase(WorkUnit, Generic[ConfigT]):
    meta: ClassVar[ComponentMeta]

    def __init__(
        self,
        *,
        rt: Runtime | object,
        config: ConfigT,
        bound_deps: Sequence[WorkUnit] | None = None,
    ) -> None:
        super().__init__(rt=cast("Runtime", rt))
        self.config = config
        self.bind_deps(*(bound_deps or ()))


__all__ = [
    "ComponentBase",
    "ComponentConfigBase",
    "ComponentMeta",
    "ConfigT",
    "DependencyRequest",
    "DependencySpec",
    "Depends",
    "InjectedParams",
    "is_dependency_request",
    "unwrap_collection_annotation",
    "unwrap_optional_annotation",
]
