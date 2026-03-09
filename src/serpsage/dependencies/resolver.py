from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import UnionType
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from serpsage.core.workunit import WorkUnit
from serpsage.dependencies.contracts import (
    _MISSING,
    BindingScope,
    InjectRequest,
    InjectToken,
    MultiBinding,
    ServiceBinding,
    ServiceKey,
    assert_service_key,
    format_service_key,
)

ParameterShape = Literal["single", "optional", "collection"]


class ServiceResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ParameterPlan:
    name: str
    annotation: Any
    default: object = inspect.Parameter.empty
    request: InjectRequest[Any] | None = None
    shape: ParameterShape = "single"
    key: ServiceKey[Any] | None = None


@dataclass(frozen=True, slots=True)
class ConstructorPlan:
    cls: type[Any]
    parameters: tuple[ParameterPlan, ...]


class ServiceCollection:
    def __init__(self) -> None:
        self._single_bindings: dict[ServiceKey[Any], ServiceBinding] = {}
        self._multi_bindings: dict[ServiceKey[Any], list[MultiBinding]] = {}

    def bind_instance(self, key: ServiceKey[Any], value: object) -> None:
        normalized = assert_service_key(key)
        self._ensure_single_free(normalized)
        self._single_bindings[normalized] = ServiceBinding(
            binding_id=f"single:{format_service_key(normalized)}",
            key=normalized,
            instance=value,
            scope=BindingScope.SINGLETON,
        )

    def bind_class(
        self,
        key: ServiceKey[Any],
        cls: type[Any],
        *,
        scope: BindingScope = BindingScope.SINGLETON,
        init_kwargs: dict[str, object] | None = None,
    ) -> None:
        normalized = assert_service_key(key)
        self._ensure_single_free(normalized)
        self._single_bindings[normalized] = ServiceBinding(
            binding_id=f"single:{format_service_key(normalized)}",
            key=normalized,
            provider_cls=cls,
            scope=scope,
            init_kwargs=dict(init_kwargs or {}),
        )

    def bind_alias(self, key: ServiceKey[Any], target_key: ServiceKey[Any]) -> None:
        normalized = assert_service_key(key)
        target = assert_service_key(target_key)
        self._ensure_single_free(normalized)
        self._single_bindings[normalized] = ServiceBinding(
            binding_id=f"single:{format_service_key(normalized)}",
            key=normalized,
            alias_key=target,
            scope=BindingScope.SINGLETON,
        )

    def bind_many(
        self,
        key: ServiceKey[Any],
        provider: type[Any] | InjectToken[Any] | object,
        *,
        order: int,
        scope: BindingScope = BindingScope.SINGLETON,
        init_kwargs: dict[str, object] | None = None,
    ) -> None:
        normalized = assert_service_key(key)
        entries = self._multi_bindings.setdefault(normalized, [])
        if any(item.order == int(order) for item in entries):
            raise ValueError(
                f"duplicate multibinding order {int(order)} for `{format_service_key(normalized)}`"
            )
        binding_id = f"multi:{format_service_key(normalized)}:{int(order)}"
        if isinstance(provider, InjectToken):
            entries.append(
                MultiBinding(
                    binding_id=binding_id,
                    key=normalized,
                    order=int(order),
                    alias_key=provider,
                    scope=scope,
                )
            )
            return
        if isinstance(provider, type):
            entries.append(
                MultiBinding(
                    binding_id=binding_id,
                    key=normalized,
                    order=int(order),
                    provider_cls=provider,
                    scope=scope,
                    init_kwargs=dict(init_kwargs or {}),
                )
            )
            return
        entries.append(
            MultiBinding(
                binding_id=binding_id,
                key=normalized,
                order=int(order),
                instance=provider,
                scope=scope,
                init_kwargs=dict(init_kwargs or {}),
            )
        )

    def build_provider(self) -> ServiceProvider:
        provider = ServiceProvider(
            single_bindings=dict(self._single_bindings),
            multi_bindings={
                key: tuple(sorted(items, key=lambda item: item.order))
                for key, items in self._multi_bindings.items()
            },
        )
        provider.validate()
        return provider

    def _ensure_single_free(self, key: ServiceKey[Any]) -> None:
        if key in self._single_bindings:
            raise ValueError(
                f"duplicate single binding for `{format_service_key(key)}`"
            )


class ServiceProvider:
    def __init__(
        self,
        *,
        single_bindings: dict[ServiceKey[Any], ServiceBinding],
        multi_bindings: dict[ServiceKey[Any], tuple[MultiBinding, ...]],
    ) -> None:
        self._single_bindings = single_bindings
        self._multi_bindings = multi_bindings
        self._plans: dict[type[Any], ConstructorPlan] = {}
        self._singletons: dict[str, object] = {}

    def require(self, key: ServiceKey[Any]) -> object:
        normalized = assert_service_key(key)
        return self._resolve_single(
            normalized,
            path=[format_service_key(normalized)],
            current_binding_id=None,
            resolving=set(),
        )

    def require_optional(self, key: ServiceKey[Any]) -> object | None:
        normalized = assert_service_key(key)
        if normalized not in self._single_bindings:
            return None
        return self.require(normalized)

    def require_many(self, key: ServiceKey[Any]) -> tuple[object, ...]:
        normalized = assert_service_key(key)
        return self._resolve_many(
            normalized,
            path=[f"{format_service_key(normalized)}[]"],
            current_binding_id=None,
            resolving=set(),
        )

    def bind_instance(self, key: ServiceKey[Any], value: object) -> None:
        normalized = assert_service_key(key)
        if normalized in self._single_bindings:
            raise ValueError(
                f"duplicate single binding for `{format_service_key(normalized)}`"
            )
        binding = ServiceBinding(
            binding_id=f"single:{format_service_key(normalized)}",
            key=normalized,
            instance=value,
            scope=BindingScope.SINGLETON,
        )
        self._single_bindings[normalized] = binding
        self._singletons[binding.binding_id] = value

    def plan_for(self, cls: type[Any]) -> ConstructorPlan:
        cached = self._plans.get(cls)
        if cached is not None:
            return cached
        plan = analyze_constructor(cls)
        self._plans[cls] = plan
        return plan

    def validate(self) -> None:
        for binding in self._single_bindings.values():
            self._validate_binding(
                binding=binding,
                path=[format_service_key(binding.key)],
                current_binding_id=None,
                visiting=set(),
            )
        for key, items in self._multi_bindings.items():
            for item in items:
                self._validate_multibinding(
                    binding=item,
                    path=[f"{format_service_key(key)}[]"],
                    current_binding_id=None,
                    visiting=set(),
                )

    def _resolve_single(
        self,
        key: ServiceKey[Any],
        *,
        path: list[str],
        current_binding_id: str | None,
        resolving: set[str],
    ) -> object:
        binding = self._single_bindings.get(key)
        if binding is None:
            raise self._resolution_error(
                f"missing binding for `{format_service_key(key)}`",
                path=path,
            )
        return self._resolve_service_binding(
            binding=binding,
            path=path,
            current_binding_id=current_binding_id,
            resolving=resolving,
        )

    def _resolve_many(
        self,
        key: ServiceKey[Any],
        *,
        path: list[str],
        current_binding_id: str | None,
        resolving: set[str],
    ) -> tuple[object, ...]:
        values: list[object] = []
        for binding in self._multi_bindings.get(key, ()):
            if (
                current_binding_id is not None
                and binding.binding_id == current_binding_id
            ):
                continue
            values.append(
                self._resolve_multi_binding(
                    binding=binding,
                    path=path,
                    current_binding_id=current_binding_id,
                    resolving=resolving,
                )
            )
        return tuple(values)

    def _resolve_service_binding(
        self,
        *,
        binding: ServiceBinding,
        path: list[str],
        current_binding_id: str | None,
        resolving: set[str],
    ) -> object:
        cache_key = binding.binding_id
        if binding.scope is BindingScope.SINGLETON and cache_key in self._singletons:
            return self._singletons[cache_key]
        if cache_key in resolving:
            raise self._resolution_error(
                f"dependency cycle detected at `{format_service_key(binding.key)}`",
                path=path,
            )
        resolving.add(cache_key)
        try:
            if binding.alias_key is not None:
                value = self._resolve_single(
                    binding.alias_key,
                    path=path + [format_service_key(binding.alias_key)],
                    current_binding_id=binding.binding_id,
                    resolving=resolving,
                )
            elif binding.instance is not _MISSING:
                value = binding.instance
            elif binding.provider_cls is not None:
                value = self._instantiate_class(
                    cls=binding.provider_cls,
                    init_kwargs=binding.init_kwargs,
                    path=path,
                    current_binding_id=binding.binding_id,
                    resolving=resolving,
                )
            else:
                raise self._resolution_error(
                    f"binding `{format_service_key(binding.key)}` has no provider",
                    path=path,
                )
        finally:
            resolving.remove(cache_key)
        if binding.scope is BindingScope.SINGLETON:
            self._singletons[cache_key] = value
        return value

    def _resolve_multi_binding(
        self,
        *,
        binding: MultiBinding,
        path: list[str],
        current_binding_id: str | None,
        resolving: set[str],
    ) -> object:
        cache_key = binding.binding_id
        if binding.scope is BindingScope.SINGLETON and cache_key in self._singletons:
            return self._singletons[cache_key]
        if cache_key in resolving:
            raise self._resolution_error(
                f"dependency cycle detected at `{binding.binding_id}`",
                path=path,
            )
        resolving.add(cache_key)
        try:
            if binding.alias_key is not None:
                value = self._resolve_single(
                    binding.alias_key,
                    path=path + [format_service_key(binding.alias_key)],
                    current_binding_id=binding.binding_id,
                    resolving=resolving,
                )
            elif binding.instance is not _MISSING:
                value = binding.instance
            elif binding.provider_cls is not None:
                value = self._instantiate_class(
                    cls=binding.provider_cls,
                    init_kwargs=binding.init_kwargs,
                    path=path,
                    current_binding_id=binding.binding_id,
                    resolving=resolving,
                )
            else:
                raise self._resolution_error(
                    f"multibinding `{binding.binding_id}` has no provider",
                    path=path,
                )
        finally:
            resolving.remove(cache_key)
        if binding.scope is BindingScope.SINGLETON:
            self._singletons[cache_key] = value
        return value

    def _instantiate_class(
        self,
        *,
        cls: type[Any],
        init_kwargs: dict[str, object],
        path: list[str],
        current_binding_id: str,
        resolving: set[str],
    ) -> object:
        plan = self.plan_for(cls)
        kwargs: dict[str, Any] = {}
        owned: list[WorkUnit] = []
        for parameter in plan.parameters:
            if parameter.name in init_kwargs:
                value = self._resolve_explicit_value(
                    init_kwargs[parameter.name],
                    parameter=parameter,
                    path=path + [f"{cls.__name__}.{parameter.name}"],
                    current_binding_id=current_binding_id,
                    resolving=resolving,
                )
            elif parameter.request is not None:
                value = self._resolve_request(
                    parameter=parameter,
                    path=path + [f"{cls.__name__}.{parameter.name}"],
                    current_binding_id=current_binding_id,
                    resolving=resolving,
                )
            elif parameter.default is not inspect.Parameter.empty:
                value = parameter.default
            else:
                raise self._resolution_error(
                    f"missing constructor argument `{cls.__name__}.{parameter.name}`",
                    path=path,
                )
            kwargs[parameter.name] = value
            self._collect_workunits(owned, value)
        instance = cls(**kwargs)
        if isinstance(instance, WorkUnit) and owned:
            instance.bind_deps(*owned)
        return instance

    def _resolve_explicit_value(
        self,
        value: object,
        *,
        parameter: ParameterPlan,
        path: list[str],
        current_binding_id: str,
        resolving: set[str],
    ) -> object:
        if isinstance(value, InjectRequest):
            key = value.key or parameter.key
            if key is None:
                raise self._resolution_error(
                    f"invalid dependency declaration for `{parameter.name}`",
                    path=path,
                )
            origin = get_origin(parameter.annotation)
            if origin in {tuple, list}:
                items = self._resolve_many(
                    key,
                    path=path,
                    current_binding_id=current_binding_id,
                    resolving=resolving,
                )
                return list(items) if origin is list else items
            target_type, optional = _unwrap_optional(parameter.annotation)
            _ = target_type
            if optional and key not in self._single_bindings:
                return None
            return self._resolve_single(
                key,
                path=path + [format_service_key(key)],
                current_binding_id=current_binding_id,
                resolving=resolving,
            )
        return value

    def _resolve_request(
        self,
        *,
        parameter: ParameterPlan,
        path: list[str],
        current_binding_id: str,
        resolving: set[str],
    ) -> object:
        if parameter.key is None:
            raise self._resolution_error(
                f"invalid dependency declaration for `{parameter.name}`",
                path=path,
            )
        if parameter.shape == "collection":
            return self._resolve_many(
                parameter.key,
                path=path,
                current_binding_id=current_binding_id,
                resolving=resolving,
            )
        if parameter.shape == "optional" and parameter.key not in self._single_bindings:
            return None
        return self._resolve_single(
            parameter.key,
            path=path + [format_service_key(parameter.key)],
            current_binding_id=current_binding_id,
            resolving=resolving,
        )

    def _validate_binding(
        self,
        *,
        binding: ServiceBinding,
        path: list[str],
        current_binding_id: str | None,
        visiting: set[str],
    ) -> None:
        cache_key = binding.binding_id
        if cache_key in visiting:
            raise self._resolution_error(
                f"dependency cycle detected at `{format_service_key(binding.key)}`",
                path=path,
            )
        if binding.instance is not _MISSING:
            return
        visiting.add(cache_key)
        try:
            if binding.alias_key is not None:
                self._validate_single_key(
                    binding.alias_key,
                    path=path + [format_service_key(binding.alias_key)],
                    current_binding_id=binding.binding_id,
                    visiting=visiting,
                )
                return
            if binding.provider_cls is None:
                raise self._resolution_error(
                    f"binding `{format_service_key(binding.key)}` has no provider",
                    path=path,
                )
            self._validate_class_plan(
                cls=binding.provider_cls,
                init_kwargs=binding.init_kwargs,
                path=path,
                current_binding_id=binding.binding_id,
                visiting=visiting,
            )
        finally:
            visiting.remove(cache_key)

    def _validate_multibinding(
        self,
        *,
        binding: MultiBinding,
        path: list[str],
        current_binding_id: str | None,
        visiting: set[str],
    ) -> None:
        cache_key = binding.binding_id
        if cache_key in visiting:
            raise self._resolution_error(
                f"dependency cycle detected at `{binding.binding_id}`",
                path=path,
            )
        if binding.instance is not _MISSING:
            return
        visiting.add(cache_key)
        try:
            if binding.alias_key is not None:
                self._validate_single_key(
                    binding.alias_key,
                    path=path + [format_service_key(binding.alias_key)],
                    current_binding_id=binding.binding_id,
                    visiting=visiting,
                )
                return
            if binding.provider_cls is None:
                raise self._resolution_error(
                    f"multibinding `{binding.binding_id}` has no provider",
                    path=path,
                )
            self._validate_class_plan(
                cls=binding.provider_cls,
                init_kwargs=binding.init_kwargs,
                path=path,
                current_binding_id=binding.binding_id,
                visiting=visiting,
            )
        finally:
            visiting.remove(cache_key)

    def _validate_class_plan(
        self,
        *,
        cls: type[Any],
        init_kwargs: dict[str, object],
        path: list[str],
        current_binding_id: str,
        visiting: set[str],
    ) -> None:
        plan = self.plan_for(cls)
        for parameter in plan.parameters:
            if parameter.name in init_kwargs:
                self._validate_explicit_value(
                    init_kwargs[parameter.name],
                    parameter=parameter,
                    path=path + [f"{cls.__name__}.{parameter.name}"],
                    current_binding_id=current_binding_id,
                    visiting=visiting,
                )
                continue
            if parameter.request is not None:
                self._validate_parameter_request(
                    parameter=parameter,
                    path=path + [f"{cls.__name__}.{parameter.name}"],
                    current_binding_id=current_binding_id,
                    visiting=visiting,
                )
                continue
            if parameter.default is inspect.Parameter.empty:
                raise self._resolution_error(
                    f"missing constructor argument `{cls.__name__}.{parameter.name}`",
                    path=path,
                )

    def _validate_explicit_value(
        self,
        value: object,
        *,
        parameter: ParameterPlan,
        path: list[str],
        current_binding_id: str,
        visiting: set[str],
    ) -> None:
        if not isinstance(value, InjectRequest):
            return
        key = value.key or parameter.key
        if key is None:
            raise self._resolution_error(
                f"invalid dependency declaration for `{parameter.name}`",
                path=path,
            )
        origin = get_origin(parameter.annotation)
        if origin in {tuple, list}:
            for binding in self._multi_bindings.get(key, ()):
                if binding.binding_id == current_binding_id:
                    continue
                self._validate_multibinding(
                    binding=binding,
                    path=path + [f"{format_service_key(key)}[]"],
                    current_binding_id=current_binding_id,
                    visiting=visiting,
                )
            return
        target_type, optional = _unwrap_optional(parameter.annotation)
        _ = target_type
        if optional and key not in self._single_bindings:
            return
        self._validate_single_key(
            key,
            path=path + [format_service_key(key)],
            current_binding_id=current_binding_id,
            visiting=visiting,
        )

    def _validate_parameter_request(
        self,
        *,
        parameter: ParameterPlan,
        path: list[str],
        current_binding_id: str,
        visiting: set[str],
    ) -> None:
        if parameter.key is None:
            raise self._resolution_error(
                f"invalid dependency declaration for `{parameter.name}`",
                path=path,
            )
        if parameter.shape == "collection":
            for binding in self._multi_bindings.get(parameter.key, ()):
                if binding.binding_id == current_binding_id:
                    continue
                self._validate_multibinding(
                    binding=binding,
                    path=path + [f"{format_service_key(parameter.key)}[]"],
                    current_binding_id=current_binding_id,
                    visiting=visiting,
                )
            return
        if parameter.shape == "optional" and parameter.key not in self._single_bindings:
            return
        self._validate_single_key(
            parameter.key,
            path=path + [format_service_key(parameter.key)],
            current_binding_id=current_binding_id,
            visiting=visiting,
        )

    def _validate_single_key(
        self,
        key: ServiceKey[Any],
        *,
        path: list[str],
        current_binding_id: str,
        visiting: set[str],
    ) -> None:
        binding = self._single_bindings.get(key)
        if binding is None:
            raise self._resolution_error(
                f"missing binding for `{format_service_key(key)}`",
                path=path,
            )
        self._validate_binding(
            binding=binding,
            path=path,
            current_binding_id=current_binding_id,
            visiting=visiting,
        )

    @staticmethod
    def _collect_workunits(out: list[WorkUnit], value: object) -> None:
        queue: list[WorkUnit] = []
        if isinstance(value, WorkUnit):
            queue.append(value)
        elif isinstance(value, tuple):
            queue.extend(item for item in value if isinstance(item, WorkUnit))
        existing = {id(item) for item in out}
        for item in queue:
            if id(item) in existing:
                continue
            existing.add(id(item))
            out.append(item)

    @staticmethod
    def _resolution_error(message: str, *, path: list[str]) -> ServiceResolutionError:
        joined = " -> ".join(token for token in path if token)
        if joined:
            return ServiceResolutionError(f"{message} (path: {joined})")
        return ServiceResolutionError(message)


def analyze_constructor(cls: type[Any]) -> ConstructorPlan:
    signature = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)
    parameters: list[ParameterPlan] = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name)
        request = (
            parameter.default
            if isinstance(parameter.default, InjectRequest)
            else None
        )
        shape = _parameter_shape(annotation)
        key = _parameter_key(annotation=annotation, request=request, shape=shape)
        if request is not None and annotation is None:
            raise TypeError(
                f"{cls.__name__}.{name} must declare a type annotation for Inject()"
            )
        if request is not None and key is None:
            raise TypeError(
                f"{cls.__name__}.{name} has an unsupported Inject() annotation"
            )
        parameters.append(
            ParameterPlan(
                name=name,
                annotation=annotation,
                default=parameter.default,
                request=request,
                shape=shape,
                key=key,
            )
        )
    return ConstructorPlan(cls=cls, parameters=tuple(parameters))


def _parameter_shape(annotation: Any) -> ParameterShape:
    collection_key, is_collection = _unwrap_collection(annotation)
    if is_collection:
        _ = collection_key
        return "collection"
    target_type, optional = _unwrap_optional(annotation)
    _ = target_type
    return "optional" if optional else "single"


def _parameter_key(
    *,
    annotation: Any,
    request: InjectRequest[Any] | None,
    shape: ParameterShape,
) -> ServiceKey[Any] | None:
    if request is not None and request.key is not None:
        return assert_service_key(request.key)
    if request is None:
        return None
    if shape == "collection":
        item_key, _ = _unwrap_collection(annotation)
        return item_key
    item_key, _optional = _unwrap_optional(annotation)
    return item_key


def _unwrap_collection(annotation: Any) -> tuple[ServiceKey[Any] | None, bool]:
    origin = get_origin(annotation)
    if origin is not tuple:
        return None, False
    args = get_args(annotation)
    if len(args) != 2 or args[1] is not Ellipsis:
        return None, True
    item = args[0]
    if isinstance(item, type):
        return item, True
    return None, True


def _unwrap_optional(annotation: Any) -> tuple[ServiceKey[Any] | None, bool]:
    origin = get_origin(annotation)
    if origin not in {UnionType, Union}:
        return (annotation if isinstance(annotation, type) else None), False
    args = [item for item in get_args(annotation) if item is not type(None)]
    if len(args) != 1:
        return None, False
    item = args[0]
    return (item if isinstance(item, type) else None), True


__all__ = [
    "ConstructorPlan",
    "ParameterPlan",
    "ServiceCollection",
    "ServiceProvider",
    "ServiceResolutionError",
    "analyze_constructor",
]
