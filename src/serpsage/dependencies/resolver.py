from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeGuard,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

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

if TYPE_CHECKING:
    from serpsage.core.workunit import WorkUnit

ParameterShape = Literal["single", "optional", "collection"]


class ServiceResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FieldPlan:
    name: str
    owner: type[Any]
    annotation: Any
    request: InjectRequest[Any]
    shape: ParameterShape = "single"
    key: ServiceKey[Any] | None = None


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


@dataclass(frozen=True, slots=True)
class ClassPlan:
    cls: type[Any]
    fields: tuple[FieldPlan, ...]
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
        overrides: dict[str, object] | None = None,
        init_kwargs: dict[str, object] | None = None,
    ) -> None:
        normalized = assert_service_key(key)
        self._ensure_single_free(normalized)
        self._single_bindings[normalized] = ServiceBinding(
            binding_id=f"single:{format_service_key(normalized)}",
            key=normalized,
            provider_cls=cls,
            scope=scope,
            overrides=_merge_overrides(overrides=overrides, init_kwargs=init_kwargs),
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
        linked_binding_id: str | None = None,
        scope: BindingScope = BindingScope.SINGLETON,
        overrides: dict[str, object] | None = None,
        init_kwargs: dict[str, object] | None = None,
    ) -> None:
        normalized = assert_service_key(key)
        entries = self._multi_bindings.setdefault(normalized, [])
        if any(item.order == int(order) for item in entries):
            raise ValueError(
                f"duplicate multibinding order {int(order)} for `{format_service_key(normalized)}`"
            )
        binding_id = f"multi:{format_service_key(normalized)}:{int(order)}"
        payload = _merge_overrides(overrides=overrides, init_kwargs=init_kwargs)
        if isinstance(provider, InjectToken):
            entries.append(
                MultiBinding(
                    binding_id=binding_id,
                    key=normalized,
                    order=int(order),
                    linked_binding_id=linked_binding_id,
                    alias_key=provider,
                    scope=scope,
                    overrides=payload,
                )
            )
            return
        if isinstance(provider, type):
            entries.append(
                MultiBinding(
                    binding_id=binding_id,
                    key=normalized,
                    order=int(order),
                    linked_binding_id=linked_binding_id,
                    provider_cls=provider,
                    scope=scope,
                    overrides=payload,
                )
            )
            return
        entries.append(
            MultiBinding(
                binding_id=binding_id,
                key=normalized,
                order=int(order),
                linked_binding_id=linked_binding_id,
                instance=provider,
                scope=scope,
                overrides=payload,
            )
        )

    def build_provider(self, *, validate: bool = True) -> ServiceProvider:
        provider = ServiceProvider(
            single_bindings=dict(self._single_bindings),
            multi_bindings={
                key: tuple(sorted(items, key=lambda item: item.order))
                for key, items in self._multi_bindings.items()
            },
        )
        if validate:
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
        self._plans: dict[type[Any], ClassPlan] = {}
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

    def plan_for(self, cls: type[Any]) -> ClassPlan:
        cached = self._plans.get(cls)
        if cached is not None:
            return cached
        plan = analyze_class(cls)
        self._plans[cls] = plan
        return plan

    def validate(self) -> None:
        for binding in self._single_bindings.values():
            self._validate_binding_entry(
                binding=binding,
                path=[format_service_key(binding.key)],
                current_binding_id=None,
                visiting=set(),
            )
        for key, items in self._multi_bindings.items():
            for item in items:
                self._validate_binding_entry(
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
        return self._resolve_binding(
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
            if current_binding_id is not None and current_binding_id in {
                binding.binding_id,
                binding.linked_binding_id,
            }:
                continue
            values.append(
                self._resolve_binding(
                    binding=binding,
                    path=path,
                    current_binding_id=current_binding_id,
                    resolving=resolving,
                )
            )
        return tuple(values)

    def _resolve_binding(
        self,
        *,
        binding: ServiceBinding | MultiBinding,
        path: list[str],
        current_binding_id: str | None,
        resolving: set[str],
    ) -> object:
        cache_key = binding.binding_id
        if binding.scope is BindingScope.SINGLETON and cache_key in self._singletons:
            return self._singletons[cache_key]
        if cache_key in resolving:
            cycle_label = (
                format_service_key(binding.key)
                if isinstance(binding, ServiceBinding)
                else binding.binding_id
            )
            raise self._resolution_error(
                f"dependency cycle detected at `{cycle_label}`",
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
                    overrides=binding.overrides,
                    path=path,
                    current_binding_id=binding.binding_id,
                    resolving=resolving,
                )
            else:
                missing_provider = (
                    f"binding `{format_service_key(binding.key)}` has no provider"
                    if isinstance(binding, ServiceBinding)
                    else f"multibinding `{binding.binding_id}` has no provider"
                )
                raise self._resolution_error(
                    missing_provider,
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
        overrides: dict[str, object],
        path: list[str],
        current_binding_id: str,
        resolving: set[str],
    ) -> object:
        plan = self.plan_for(cls)
        self._validate_override_targets(plan=plan, overrides=overrides, path=path)
        resolved_fields: dict[str, object] = {}
        resolved_params: dict[str, object] = {}
        owned: list[WorkUnit] = []
        instance = object.__new__(cls)

        if _is_workunit_instance(instance):
            runtime_value = self._resolve_workunit_runtime(
                plan=plan,
                overrides=overrides,
                resolved_fields=resolved_fields,
                resolved_params=resolved_params,
                path=path,
                current_binding_id=current_binding_id,
                resolving=resolving,
            )
            _bootstrap_workunit(instance, runtime_value)
        if _is_component_subclass(cls):
            config_value = overrides.get("config", _MISSING)
            if config_value is _MISSING:
                raise self._resolution_error(
                    f"`{cls.__name__}` is missing component config",
                    path=path,
                )
            _bind_component_config(instance, config_value)

        for field in plan.fields:
            value = self._resolve_member_value(
                member=field,
                overrides=overrides,
                resolved=resolved_fields,
                path=path + [f"{cls.__name__}.{field.name}"],
                current_binding_id=current_binding_id,
                resolving=resolving,
            )
            setattr(instance, field.name, value)
            self._collect_workunits(owned, value)

        kwargs: dict[str, object] = {}
        for parameter in plan.parameters:
            value = self._resolve_member_value(
                member=parameter,
                overrides=overrides,
                resolved=resolved_params,
                path=path + [f"{cls.__name__}.{parameter.name}"],
                current_binding_id=current_binding_id,
                resolving=resolving,
            )
            kwargs[parameter.name] = value
            self._collect_workunits(owned, value)

        cls.__init__(instance, **kwargs)
        if _is_workunit_instance(instance) and owned:
            instance.bind_deps(*owned)
        return instance

    def _resolve_workunit_runtime(
        self,
        *,
        plan: ClassPlan,
        overrides: dict[str, object],
        resolved_fields: dict[str, object],
        resolved_params: dict[str, object],
        path: list[str],
        current_binding_id: str,
        resolving: set[str],
    ) -> object:
        field = next((item for item in plan.fields if item.name == "rt"), None)
        if field is not None:
            return self._resolve_member_value(
                member=field,
                overrides=overrides,
                resolved=resolved_fields,
                path=path + [f"{plan.cls.__name__}.rt"],
                current_binding_id=current_binding_id,
                resolving=resolving,
            )
        parameter = next((item for item in plan.parameters if item.name == "rt"), None)
        if parameter is not None:
            return self._resolve_member_value(
                member=parameter,
                overrides=overrides,
                resolved=resolved_params,
                path=path + [f"{plan.cls.__name__}.rt"],
                current_binding_id=current_binding_id,
                resolving=resolving,
            )
        raise self._resolution_error(
            f"`{plan.cls.__name__}` is a WorkUnit but has no runtime dependency source",
            path=path,
        )

    def _resolve_member_value(
        self,
        *,
        member: FieldPlan | ParameterPlan,
        overrides: dict[str, object],
        resolved: dict[str, object],
        path: list[str],
        current_binding_id: str,
        resolving: set[str],
    ) -> object:
        cached = resolved.get(member.name, _MISSING)
        if cached is not _MISSING:
            return cached
        if member.name in overrides:
            override_value = overrides[member.name]
            if not isinstance(override_value, InjectRequest):
                value = override_value
            else:
                target_key = override_value.key or member.key
                if target_key is None:
                    raise self._resolution_error(
                        f"invalid dependency declaration for `{member.name}`",
                        path=path,
                    )
                origin = get_origin(member.annotation)
                if origin in {tuple, list}:
                    value = self._resolve_dependency(
                        shape="collection",
                        key=target_key,
                        name=member.name,
                        path=path,
                        current_binding_id=current_binding_id,
                        resolving=resolving,
                        as_list=origin is list,
                    )
                else:
                    annotation_shape, _ = _analyze_annotation(member.annotation)
                    value = self._resolve_dependency(
                        shape=annotation_shape,
                        key=target_key,
                        name=member.name,
                        path=path,
                        current_binding_id=current_binding_id,
                        resolving=resolving,
                    )
        elif member.request is not None:
            value = self._resolve_dependency(
                shape=member.shape,
                key=member.key,
                name=member.name,
                path=path,
                current_binding_id=current_binding_id,
                resolving=resolving,
            )
        elif (
            isinstance(member, ParameterPlan)
            and member.default is not inspect.Parameter.empty
        ):
            value = member.default
        else:
            raise self._resolution_error(
                f"missing constructor argument `{member.name}`",
                path=path,
            )
        resolved[member.name] = value
        return value

    def _resolve_dependency(
        self,
        *,
        shape: ParameterShape,
        key: ServiceKey[Any] | None,
        name: str,
        path: list[str],
        current_binding_id: str,
        resolving: set[str],
        as_list: bool = False,
    ) -> object:
        if key is None:
            raise self._resolution_error(
                f"invalid dependency declaration for `{name}`",
                path=path,
            )
        if shape == "collection":
            items = self._resolve_many(
                key,
                path=path,
                current_binding_id=current_binding_id,
                resolving=resolving,
            )
            return list(items) if as_list else items
        if shape == "optional" and key not in self._single_bindings:
            return None
        return self._resolve_single(
            key,
            path=path + [format_service_key(key)],
            current_binding_id=current_binding_id,
            resolving=resolving,
        )

    def _validate_binding_entry(
        self,
        *,
        binding: ServiceBinding | MultiBinding,
        path: list[str],
        current_binding_id: str | None,
        visiting: set[str],
    ) -> None:
        cache_key = binding.binding_id
        if cache_key in visiting:
            cycle_label = (
                format_service_key(binding.key)
                if isinstance(binding, ServiceBinding)
                else binding.binding_id
            )
            raise self._resolution_error(
                f"dependency cycle detected at `{cycle_label}`",
                path=path,
            )
        if binding.instance is not _MISSING:
            return
        visiting.add(cache_key)
        try:
            if binding.alias_key is not None:
                self._validate_dependency(
                    shape="single",
                    key=binding.alias_key,
                    name=format_service_key(binding.alias_key),
                    path=path + [format_service_key(binding.alias_key)],
                    current_binding_id=binding.binding_id,
                    visiting=visiting,
                )
                return
            if binding.provider_cls is None:
                missing_provider = (
                    f"binding `{format_service_key(binding.key)}` has no provider"
                    if isinstance(binding, ServiceBinding)
                    else f"multibinding `{binding.binding_id}` has no provider"
                )
                raise self._resolution_error(
                    missing_provider,
                    path=path,
                )
            self._validate_class_plan(
                cls=binding.provider_cls,
                overrides=binding.overrides,
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
        overrides: dict[str, object],
        path: list[str],
        current_binding_id: str,
        visiting: set[str],
    ) -> None:
        plan = self.plan_for(cls)
        self._validate_override_targets(plan=plan, overrides=overrides, path=path)
        if _is_workunit_subclass(cls):
            self._validate_workunit_runtime_source(
                plan=plan,
                overrides=overrides,
                path=path,
                current_binding_id=current_binding_id,
                visiting=visiting,
            )
        for field in plan.fields:
            self._validate_member(
                member=field,
                overrides=overrides,
                path=path + [f"{cls.__name__}.{field.name}"],
                current_binding_id=current_binding_id,
                visiting=visiting,
            )
        for parameter in plan.parameters:
            self._validate_member(
                member=parameter,
                overrides=overrides,
                path=path + [f"{cls.__name__}.{parameter.name}"],
                current_binding_id=current_binding_id,
                visiting=visiting,
            )

    def _validate_member(
        self,
        *,
        member: FieldPlan | ParameterPlan,
        overrides: dict[str, object],
        path: list[str],
        current_binding_id: str,
        visiting: set[str],
    ) -> None:
        if member.name in overrides:
            override_value = overrides[member.name]
            if not isinstance(override_value, InjectRequest):
                return
            target_key = override_value.key or member.key
            if target_key is None:
                raise self._resolution_error(
                    f"invalid dependency declaration for `{member.name}`",
                    path=path,
                )
            origin = get_origin(member.annotation)
            if origin in {tuple, list}:
                self._validate_dependency(
                    shape="collection",
                    key=target_key,
                    name=member.name,
                    path=path,
                    current_binding_id=current_binding_id,
                    visiting=visiting,
                )
                return
            annotation_shape, _ = _analyze_annotation(member.annotation)
            self._validate_dependency(
                shape=annotation_shape,
                key=target_key,
                name=member.name,
                path=path,
                current_binding_id=current_binding_id,
                visiting=visiting,
            )
            return
        if member.request is not None:
            self._validate_dependency(
                shape=member.shape,
                key=member.key,
                name=member.name,
                path=path,
                current_binding_id=current_binding_id,
                visiting=visiting,
            )
            return
        if (
            isinstance(member, ParameterPlan)
            and member.default is inspect.Parameter.empty
        ):
            raise self._resolution_error(
                f"missing constructor argument `{member.name}`",
                path=path,
            )

    def _validate_dependency(
        self,
        *,
        shape: ParameterShape,
        key: ServiceKey[Any] | None,
        name: str,
        path: list[str],
        current_binding_id: str,
        visiting: set[str],
    ) -> None:
        if key is None:
            raise self._resolution_error(
                f"invalid dependency declaration for `{name}`",
                path=path,
            )
        if shape == "collection":
            for binding in self._multi_bindings.get(key, ()):
                if current_binding_id in {
                    binding.binding_id,
                    binding.linked_binding_id,
                }:
                    continue
                self._validate_binding_entry(
                    binding=binding,
                    path=path + [f"{format_service_key(key)}[]"],
                    current_binding_id=current_binding_id,
                    visiting=visiting,
                )
            return
        if shape == "optional" and key not in self._single_bindings:
            return
        single_binding = self._single_bindings.get(key)
        if single_binding is None:
            raise self._resolution_error(
                f"missing binding for `{format_service_key(key)}`",
                path=path,
            )
        self._validate_binding_entry(
            binding=single_binding,
            path=path,
            current_binding_id=current_binding_id,
            visiting=visiting,
        )

    def _validate_workunit_runtime_source(
        self,
        *,
        plan: ClassPlan,
        overrides: dict[str, object],
        path: list[str],
        current_binding_id: str,
        visiting: set[str],
    ) -> None:
        field = next((item for item in plan.fields if item.name == "rt"), None)
        if field is not None:
            self._validate_member(
                member=field,
                overrides=overrides,
                path=path + [f"{plan.cls.__name__}.rt"],
                current_binding_id=current_binding_id,
                visiting=visiting,
            )
            return
        parameter = next((item for item in plan.parameters if item.name == "rt"), None)
        if parameter is not None:
            self._validate_member(
                member=parameter,
                overrides=overrides,
                path=path + [f"{plan.cls.__name__}.rt"],
                current_binding_id=current_binding_id,
                visiting=visiting,
            )
            return
        raise self._resolution_error(
            f"`{plan.cls.__name__}` is a WorkUnit but has no runtime dependency source",
            path=path,
        )

    @staticmethod
    def _validate_override_targets(
        *,
        plan: ClassPlan,
        overrides: dict[str, object],
        path: list[str],
    ) -> None:
        valid_names = {item.name for item in plan.fields}
        valid_names.update(item.name for item in plan.parameters)
        if _is_component_subclass(plan.cls):
            valid_names.add("config")
        unknown = sorted(name for name in overrides if name not in valid_names)
        if unknown:
            joined = ", ".join(f"`{name}`" for name in unknown)
            raise ServiceResolutionError(
                f"invalid override target(s) {joined} for `{plan.cls.__name__}` "
                f"(path: {' -> '.join(path)})"
            )

    @staticmethod
    def _collect_workunits(out: list[WorkUnit], value: object) -> None:
        queue: list[WorkUnit] = []
        if _is_workunit_instance(value):
            queue.append(value)
        elif isinstance(value, (tuple, list)):
            queue.extend(item for item in value if _is_workunit_instance(item))
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


def _workunit_class() -> type[WorkUnit]:
    from serpsage.core.workunit import WorkUnit

    return WorkUnit


def _bootstrap_workunit(instance: WorkUnit, rt: object) -> None:
    from serpsage.core.workunit import bootstrap_workunit

    bootstrap_workunit(instance, rt)


def _component_base_class() -> type[Any]:
    from serpsage.components.base import ComponentBase

    return ComponentBase


def _bind_component_config(instance: object, config: object) -> None:
    cast("Any", instance)._component_config = config


def _is_workunit_instance(value: object) -> TypeGuard[WorkUnit]:
    return isinstance(value, _workunit_class())


def _is_workunit_subclass(cls: type[Any]) -> TypeGuard[type[WorkUnit]]:
    return issubclass(cls, _workunit_class())


def _is_component_subclass(cls: type[Any]) -> bool:
    return issubclass(cls, _component_base_class())


def analyze_class(cls: type[Any]) -> ClassPlan:
    parameters = _analyze_parameters(cls)
    parameter_names = {item.name for item in parameters}
    fields_by_name: dict[str, FieldPlan] = {}
    for owner in reversed(cls.__mro__[:-1]):
        annotations = _get_class_hints(owner)
        for name, value in owner.__dict__.items():
            if not isinstance(value, InjectRequest):
                continue
            if name in parameter_names and owner is not cls:
                continue
            if name in parameter_names and owner is cls:
                raise TypeError(
                    f"{cls.__name__}.{name} cannot be both an injected field and a constructor parameter"
                )
            annotation = annotations.get(name)
            if annotation is None:
                raise TypeError(
                    f"{owner.__name__}.{name} must declare a type annotation for Inject()"
                )
            shape, key = _member_contract(annotation=annotation, request=value)
            if key is None:
                raise TypeError(
                    f"{owner.__name__}.{name} has an unsupported Inject() annotation"
                )
            fields_by_name[name] = FieldPlan(
                name=name,
                owner=owner,
                annotation=annotation,
                request=value,
                shape=shape,
                key=key,
            )
    return ClassPlan(
        cls=cls,
        fields=tuple(fields_by_name.values()),
        parameters=parameters,
    )


def analyze_constructor(cls: type[Any]) -> ConstructorPlan:
    plan = analyze_class(cls)
    return ConstructorPlan(cls=cls, parameters=plan.parameters)


def _analyze_parameters(cls: type[Any]) -> tuple[ParameterPlan, ...]:
    if _uses_inherited_compat_init(cls):
        return ()
    signature = inspect.signature(cls.__init__)
    hints = _get_function_hints(cls.__init__)
    parameters: list[ParameterPlan] = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name)
        request = (
            parameter.default if isinstance(parameter.default, InjectRequest) else None
        )
        if request is not None and annotation is None:
            raise TypeError(
                f"{cls.__name__}.{name} must declare a type annotation for Inject()"
            )
        shape, key = _member_contract(annotation=annotation, request=request)
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
    return tuple(parameters)


def _member_contract(
    *,
    annotation: Any,
    request: InjectRequest[Any] | None,
) -> tuple[ParameterShape, ServiceKey[Any] | None]:
    shape, inferred_key = _analyze_annotation(annotation)
    if request is None:
        return shape, None
    if request.key is not None:
        return shape, assert_service_key(request.key)
    return shape, inferred_key


def _analyze_annotation(
    annotation: Any,
) -> tuple[ParameterShape, ServiceKey[Any] | None]:
    origin = get_origin(annotation)
    if origin is list:
        list_args = get_args(annotation)
        if len(list_args) != 1:
            return "collection", None
        return "collection", _service_key_from_annotation(list_args[0])
    if origin is tuple:
        tuple_args = get_args(annotation)
        if len(tuple_args) != 2 or tuple_args[1] is not Ellipsis:
            return "collection", None
        return "collection", _service_key_from_annotation(tuple_args[0])
    if origin in {UnionType, Union}:
        union_args = [item for item in get_args(annotation) if item is not type(None)]
        if len(union_args) != 1:
            return "single", None
        return "optional", _service_key_from_annotation(union_args[0])
    return "single", _service_key_from_annotation(annotation)


def _service_key_from_annotation(annotation: Any) -> ServiceKey[Any] | None:
    if isinstance(annotation, type):
        return annotation
    origin = get_origin(annotation)
    return origin if isinstance(origin, type) else None


def _merge_overrides(
    *,
    overrides: dict[str, object] | None,
    init_kwargs: dict[str, object] | None,
) -> dict[str, object]:
    if overrides and init_kwargs:
        raise ValueError("pass either `overrides` or `init_kwargs`, not both")
    return dict(overrides or init_kwargs or {})


def _uses_inherited_compat_init(cls: type[Any]) -> bool:
    if "__init__" in cls.__dict__:
        return False
    if cls.__init__ is object.__init__:
        return True
    if cls.__init__ is _workunit_class().__init__:
        return True
    try:
        from serpsage.components.base import ComponentBase
    except Exception:  # noqa: BLE001
        return False
    return cls.__init__ is ComponentBase.__init__


def _get_function_hints(func: Any) -> dict[str, Any]:
    return get_type_hints(func, globalns=_annotation_globals(func), localns={})


def _get_class_hints(cls: type[Any]) -> dict[str, Any]:
    return inspect.get_annotations(
        cls,
        globals=_annotation_globals(cls),
        locals=dict(vars(cls)),
        eval_str=True,
    )


def _annotation_globals(obj: Any) -> dict[str, Any]:
    module = sys.modules.get(getattr(obj, "__module__", ""))
    globalns = dict(vars(module)) if module is not None else {}
    runtime_type: object | None = None
    if "Runtime" not in globalns:
        try:
            from serpsage.core.runtime import Runtime
        except Exception:  # noqa: BLE001
            runtime_type = None
        else:
            runtime_type = Runtime
        if runtime_type is not None:
            globalns["Runtime"] = runtime_type
    return globalns


__all__ = [
    "ClassPlan",
    "ConstructorPlan",
    "FieldPlan",
    "ParameterPlan",
    "ServiceCollection",
    "ServiceProvider",
    "ServiceResolutionError",
    "analyze_class",
    "analyze_constructor",
]
