from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast, get_type_hints, overload

import httpx

from serpsage.components.base import (
    ComponentConfigBase,
    DependencyRequest,
    InjectedParams,
    is_dependency_request,
    unwrap_collection_annotation,
    unwrap_optional_annotation,
)
from serpsage.components.registry import ComponentDescriptor, ComponentRegistry
from serpsage.core.workunit import WorkUnit
from serpsage.settings.models import AppSettings, ComponentFamilySettings

if TYPE_CHECKING:
    from serpsage.core.runtime import Overrides, Runtime

TWorkUnit = TypeVar("TWorkUnit", bound=WorkUnit)
TObject = TypeVar("TObject")
TConfig = TypeVar("TConfig", bound=ComponentConfigBase)


class ComponentResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResolvedComponentSpec:
    family: str
    instance_id: str
    component_name: str
    descriptor: ComponentDescriptor
    config: ComponentConfigBase


@dataclass(frozen=True, slots=True)
class _TypeCandidate:
    family: str
    instance_id: str
    priority: int
    is_default: bool
    spec: ResolvedComponentSpec | None = None
    value: WorkUnit | None = None


class ComponentContainer:
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
        self._runtime: Runtime | None = None
        self._spec_cache: dict[tuple[str, str], ResolvedComponentSpec] = {}
        self._instance_cache: dict[tuple[str, str], WorkUnit] = {}
        self._build_stack: list[tuple[str, str]] = []

    def attach_runtime(self, rt: Runtime) -> None:
        self._runtime = rt

    def resolve_default(
        self,
        family: str,
        *,
        expected_type: type[TWorkUnit] | None = None,
    ) -> TWorkUnit:
        override = self._resolve_override(family)
        if override is not None:
            return self._coerce_expected(override, expected_type=expected_type)
        family_settings = self.family_settings(family)
        return self.resolve_instance(
            family=family,
            instance_id=family_settings.default,
            expected_type=expected_type,
        )

    def resolve_default_spec(self, family: str) -> ResolvedComponentSpec:
        family_settings = self.family_settings(family)
        return self.resolve_spec(family=family, instance_id=family_settings.default)

    @overload
    def resolve_default_config(
        self,
        family: str,
        *,
        expected_type: type[TConfig],
    ) -> TConfig: ...

    @overload
    def resolve_default_config(
        self,
        family: str,
        *,
        expected_type: None = None,
    ) -> ComponentConfigBase: ...

    def resolve_default_config(
        self,
        family: str,
        *,
        expected_type: type[TConfig] | None = None,
    ) -> ComponentConfigBase | TConfig:
        config = self.resolve_default_spec(family).config
        if expected_type is None:
            return config
        if not isinstance(config, expected_type):
            raise TypeError(
                f"default config for `{family}` expected `{expected_type.__name__}`, "
                f"got `{type(config).__name__}`"
            )
        return config

    def resolve_instance(
        self,
        *,
        family: str,
        instance_id: str,
        expected_type: type[TWorkUnit] | None = None,
    ) -> TWorkUnit:
        override = self._resolve_override(family)
        if override is not None and instance_id == self.family_settings(family).default:
            return self._coerce_expected(override, expected_type=expected_type)
        key = (str(family), str(instance_id))
        cached = self._instance_cache.get(key)
        if cached is not None:
            return self._coerce_expected(cached, expected_type=expected_type)
        spec = self.resolve_spec(family=family, instance_id=instance_id)
        if key in self._build_stack:
            raise ComponentResolutionError(
                f"dependency cycle detected while building `{family}:{instance_id}`"
            )
        self._build_stack.append(key)
        try:
            instance = self.instantiate_component(spec)
            self._instance_cache[key] = instance
        finally:
            self._build_stack.pop()
        return self._coerce_expected(instance, expected_type=expected_type)

    def resolve_spec(self, *, family: str, instance_id: str) -> ResolvedComponentSpec:
        key = (str(family), str(instance_id))
        cached = self._spec_cache.get(key)
        if cached is not None:
            return cached
        family_settings = self.family_settings(family)
        instance_settings = family_settings.instances.get(instance_id)
        if instance_settings is None:
            raise ComponentResolutionError(
                f"component instance `{family}:{instance_id}` does not exist"
            )
        if not bool(instance_settings.enabled):
            raise ComponentResolutionError(
                f"component instance `{family}:{instance_id}` is disabled"
            )
        descriptor = self._registry.get(family, instance_settings.component)
        config = descriptor.meta.config_model.from_raw(
            dict(instance_settings.config or {}),
            env=self._env,
        )
        spec = ResolvedComponentSpec(
            family=str(family),
            instance_id=str(instance_id),
            component_name=str(instance_settings.component),
            descriptor=descriptor,
            config=config,
        )
        self._spec_cache[key] = spec
        return spec

    def instantiate_component(self, spec: ResolvedComponentSpec) -> WorkUnit:
        injected = self._resolve_constructor_params(
            cls=spec.descriptor.cls,
            current_config=spec.config,
            current_key=(spec.family, spec.instance_id),
        )
        return spec.descriptor.cls(**injected.kwargs)

    def instantiate_class(
        self,
        cls: type[TObject],
        *,
        explicit_kwargs: dict[str, Any] | None = None,
    ) -> TObject:
        injected = self._resolve_constructor_params(
            cls=cls,
            current_config=None,
            current_key=None,
            explicit_kwargs=explicit_kwargs,
        )
        return cls(**injected.kwargs)

    def family_settings(self, family: str) -> ComponentFamilySettings:
        value = getattr(self._settings, str(family or ""), None)
        if not isinstance(value, ComponentFamilySettings):
            raise ComponentResolutionError(
                f"settings family `{family}` is not configured as a component family"
            )
        return value

    def family_name(self, family: str) -> str:
        return self.resolve_default_spec(family).component_name

    def http_override(self) -> httpx.AsyncClient | None:
        ov = self._overrides
        if ov is None:
            return None
        client = ov.http
        return client if isinstance(client, httpx.AsyncClient) else None

    def _resolve_constructor_params(
        self,
        *,
        cls: type[Any],
        current_config: ComponentConfigBase | None,
        current_key: tuple[str, str] | None,
        explicit_kwargs: dict[str, Any] | None = None,
    ) -> InjectedParams:
        signature = inspect.signature(cls.__init__)
        hints = get_type_hints(cls.__init__)
        kwargs: dict[str, Any] = {}
        bound_deps: list[WorkUnit] = []
        local_cache: dict[tuple[str, str | None, bool], object] = {}
        provided = dict(explicit_kwargs or {})
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if name in provided:
                kwargs[name] = provided[name]
                continue
            if name == "rt":
                kwargs[name] = self._require_runtime()
                continue
            if name == "config" and current_config is not None:
                kwargs[name] = current_config
                continue
            annotation = hints.get(name)
            framework_value = self._resolve_framework_arg(
                annotation=annotation,
                current_config=current_config,
            )
            if framework_value is not _UNSET:
                kwargs[name] = framework_value
                continue
            default = parameter.default
            if is_dependency_request(default):
                request = cast("DependencyRequest", default)
                if annotation is None:
                    raise ComponentResolutionError(
                        f"{cls.__name__}.{name} requires a type annotation for Depends()"
                    )
                cache_key = (
                    repr(annotation),
                    request.instance,
                    request.optional,
                )
                if request.use_cache and cache_key in local_cache:
                    value = local_cache[cache_key]
                else:
                    value = self._resolve_typed_dependency(
                        owner=cls,
                        annotation=annotation,
                        request=request,
                        current_key=current_key,
                    )
                    if request.use_cache:
                        local_cache[cache_key] = value
                kwargs[name] = value
                self._extend_bound_deps(bound_deps=bound_deps, value=value)
                continue
            if parameter.default is not inspect.Parameter.empty:
                kwargs[name] = parameter.default
                continue
            raise ComponentResolutionError(
                f"cannot resolve constructor argument `{cls.__name__}.{name}`"
            )
        if current_config is not None:
            kwargs.setdefault("bound_deps", tuple(bound_deps))
        return InjectedParams(kwargs=kwargs, bound_deps=tuple(bound_deps))

    def _resolve_framework_arg(
        self,
        *,
        annotation: Any,
        current_config: ComponentConfigBase | None,
    ) -> object:
        if annotation is None:
            return _UNSET
        rt = self._require_runtime()
        if annotation is type(rt):
            return rt
        if annotation is AppSettings:
            return self._settings
        if annotation is ComponentContainer:
            return self
        if (
            current_config is not None
            and isinstance(annotation, type)
            and issubclass(annotation, ComponentConfigBase)
            and isinstance(current_config, annotation)
        ):
            return current_config
        return _UNSET

    def _resolve_typed_dependency(
        self,
        *,
        owner: type[Any],
        annotation: Any,
        request: DependencyRequest,
        current_key: tuple[str, str] | None,
    ) -> object:
        collection_origin, item_type = unwrap_collection_annotation(annotation)
        if collection_origin is not None:
            if item_type is None:
                raise ComponentResolutionError(
                    f"{owner.__name__} collection dependency must declare its item type"
                )
            candidates = self._matching_type_candidates(
                requested_type=item_type,
                current_key=current_key,
            )
            ordered = [self._materialize_candidate(item) for item in candidates]
            if collection_origin is tuple:
                return tuple(ordered)
            return ordered
        resolved_type = unwrap_optional_annotation(annotation)
        if resolved_type is None:
            raise ComponentResolutionError(
                f"{owner.__name__} dependency annotations must be concrete classes"
            )
        candidates = self._matching_type_candidates(
            requested_type=resolved_type,
            current_key=current_key,
        )
        if request.instance:
            candidates = self._filter_candidates_by_instance(
                candidates=candidates,
                selector=request.instance,
            )
        if not candidates:
            if request.optional:
                return None
            raise ComponentResolutionError(
                f"{owner.__name__} requires dependency `{resolved_type.__name__}`"
            )
        if len(candidates) == 1:
            return self._materialize_candidate(candidates[0])
        default_candidates = [item for item in candidates if item.is_default]
        if len(default_candidates) == 1:
            return self._materialize_candidate(default_candidates[0])
        raise ComponentResolutionError(
            f"ambiguous dependency for `{owner.__name__}` with type `{resolved_type.__name__}`"
        )

    def _matching_type_candidates(
        self,
        *,
        requested_type: type[Any],
        current_key: tuple[str, str] | None,
    ) -> list[_TypeCandidate]:
        matches: list[_TypeCandidate] = []
        overridden_families = self._override_families()
        for family, override in overridden_families.items():
            if isinstance(override, requested_type):
                matches.append(
                    _TypeCandidate(
                        family=family,
                        instance_id=self.family_settings(family).default,
                        priority=-1,
                        is_default=True,
                        value=override,
                    )
                )
        for family in self._component_families():
            if family in overridden_families:
                continue
            family_settings = self.family_settings(family)
            default_instance_id = str(family_settings.default)
            for instance_id, instance_settings in family_settings.instances.items():
                key = (family, str(instance_id))
                if current_key is not None and key == current_key:
                    continue
                if not bool(instance_settings.enabled):
                    continue
                descriptor = self._registry.get(family, instance_settings.component)
                if not issubclass(descriptor.cls, requested_type):
                    continue
                spec = self.resolve_spec(family=family, instance_id=str(instance_id))
                matches.append(
                    _TypeCandidate(
                        family=family,
                        instance_id=str(instance_id),
                        priority=int(spec.descriptor.meta.priority),
                        is_default=str(instance_id) == default_instance_id,
                        spec=spec,
                    )
                )
        matches.sort(
            key=lambda item: (
                0 if item.is_default else 1,
                int(item.priority),
                item.family,
                item.instance_id,
            )
        )
        return matches

    def _filter_candidates_by_instance(
        self,
        *,
        candidates: list[_TypeCandidate],
        selector: str,
    ) -> list[_TypeCandidate]:
        family: str | None = None
        instance_id = str(selector)
        if ":" in instance_id:
            family, instance_id = instance_id.split(":", 1)
        return [
            item
            for item in candidates
            if item.instance_id == instance_id
            and (family is None or item.family == family)
        ]

    def _materialize_candidate(self, candidate: _TypeCandidate) -> WorkUnit:
        if candidate.value is not None:
            return candidate.value
        spec = candidate.spec
        if spec is None:
            raise ComponentResolutionError(
                "dependency candidate is missing its component spec"
            )
        return self.resolve_instance(
            family=spec.family,
            instance_id=spec.instance_id,
        )

    def _override_families(self) -> dict[str, WorkUnit]:
        ov = self._overrides
        if ov is None:
            return {}
        mapping = {
            "provider": ov.provider,
            "fetch": ov.fetcher,
            "extract": ov.extractor,
            "rank": ov.ranker,
            "llm": ov.llm,
            "cache": ov.cache,
            "telemetry": ov.telemetry,
            "rate_limit": ov.rate_limiter,
        }
        return {
            family: candidate
            for family, candidate in mapping.items()
            if isinstance(candidate, WorkUnit)
        }

    def _component_families(self) -> tuple[str, ...]:
        return (
            "http",
            "provider",
            "fetch",
            "extract",
            "rank",
            "llm",
            "cache",
            "telemetry",
            "rate_limit",
        )

    def _resolve_override(self, family: str) -> WorkUnit | None:
        return self._override_families().get(str(family))

    def _require_runtime(self) -> Runtime:
        rt = self._runtime
        if rt is None:
            raise ComponentResolutionError(
                "component container runtime is not attached"
            )
        return rt

    def _extend_bound_deps(self, *, bound_deps: list[WorkUnit], value: object) -> None:
        if isinstance(value, WorkUnit):
            bound_deps.append(value)
            return
        if isinstance(value, tuple):
            bound_deps.extend(item for item in value if isinstance(item, WorkUnit))
            return
        if isinstance(value, list):
            bound_deps.extend(item for item in value if isinstance(item, WorkUnit))

    def _coerce_expected(
        self,
        value: WorkUnit,
        *,
        expected_type: type[TWorkUnit] | None,
    ) -> TWorkUnit:
        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeError(
                f"component expected `{expected_type.__name__}`, got `{type(value).__name__}`"
            )
        return cast("TWorkUnit", value)


class _Unset:
    pass


_UNSET = _Unset()


__all__ = [
    "ComponentContainer",
    "ComponentResolutionError",
    "ResolvedComponentSpec",
]
