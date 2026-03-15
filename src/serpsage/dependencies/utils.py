from __future__ import annotations

import asyncio
import inspect
import os
import sys
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
)
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    asynccontextmanager,
    contextmanager,
)
from contextlib import AbstractContextManager as ContextManager
from functools import partial
from inspect import get_annotations
from typing import Any, TypeAlias, TypeVar, Union, cast, get_origin, get_type_hints
from typing_extensions import ParamSpec, override

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")

StrOrBytesPath = Union[str, bytes, os.PathLike[Any]]  # type alias  # noqa: UP007
TreeType = dict[_T, Union[Any, "TreeType[_T]"]]

Dependency: TypeAlias = (
    type[_T | AbstractAsyncContextManager[_T] | AbstractContextManager[_T]]
    | Callable[..., _T]
    | Callable[..., Awaitable[_T]]
    | str
)


class InnerDepends:
    dependency: Dependency[Any] | None
    use_cache: bool

    def __init__(
        self, dependency: Dependency[Any] | None = None, *, use_cache: bool = True
    ) -> None:
        self.dependency = dependency
        self.use_cache = use_cache

    @override
    def __repr__(self) -> str:
        attr = getattr(self.dependency, "__name__", type(self.dependency).__name__)
        cache = "" if self.use_cache else ", use_cache=False"
        return f"InnerDepends({attr}{cache})"


def get_dependency_name(dependency: Dependency[Any]) -> str:
    if isinstance(dependency, str):
        return dependency
    if isinstance(dependency, type):
        return dependency.__name__
    if callable(dependency):
        if hasattr(dependency, "__name__"):
            return (
                dependency.__name__ if dependency.__name__ != "<lambda>" else "lambda"
            )
        return dependency.__class__.__name__
    return dependency.__class__.__name__


def _normalize_dependency_target(target: Any) -> Any:
    origin = get_origin(target)
    if isinstance(origin, type):
        return origin
    return target


def sync_func_wrapper(
    func: Callable[_P, _R], *, to_thread: bool = False
) -> Callable[_P, Coroutine[None, None, _R]]:
    if to_thread:

        async def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            loop = asyncio.get_running_loop()
            func_call = partial(func, *args, **kwargs)
            return await loop.run_in_executor(None, func_call)

    else:

        async def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return func(*args, **kwargs)

    return _wrapper


@asynccontextmanager
async def sync_ctx_manager_wrapper(
    cm: ContextManager[_T], *, to_thread: bool = False
) -> AsyncGenerator[_T, None]:
    try:
        yield await sync_func_wrapper(cm.__enter__, to_thread=to_thread)()
    except Exception as e:
        if not await sync_func_wrapper(cm.__exit__, to_thread=to_thread)(
            type(e), e, e.__traceback__
        ):
            raise
    else:
        await sync_func_wrapper(cm.__exit__, to_thread=to_thread)(None, None, None)


async def _execute_callable(
    dependent: Callable[..., Any],
    stack: AsyncExitStack | None,
    dependency_cache: dict[Any, Any],
) -> Any:
    func_args = await _resolve_callable_args(dependent, stack, dependency_cache)
    if inspect.iscoroutinefunction(dependent):
        return await dependent(**func_args)
    return dependent(**func_args)


async def _resolve_callable_args(
    dependent: Callable[..., Any],
    stack: AsyncExitStack | None,
    dependency_cache: dict[Any, Any],
) -> dict[str, Any]:
    func_params = inspect.signature(dependent).parameters
    func_args: dict[str, Any] = {}

    for param_name, param in func_params.items():
        try:
            param_type = get_type_hints(dependent).get(param_name)
        except NameError:
            param_type = param.annotation
        normalized_param_type = _normalize_dependency_target(param_type)
        if isinstance(param.default, InnerDepends):
            target_dependency = param.default.dependency
            if target_dependency is None:
                target_dependency = cast(
                    "Dependency[Any] | None", normalized_param_type
                )
            if target_dependency is None:
                raise TypeError(
                    f"Cannot resolve parameter '{param_name}' for dependency "
                    f"'{dependent.__name__}' without an explicit dependency or annotation"
                )
            func_args[param_name] = await solve_dependencies(
                target_dependency,
                use_cache=param.default.use_cache,
                stack=stack,
                dependency_cache=dependency_cache,
            )
        elif param.default is not inspect.Parameter.empty:
            func_args[param_name] = param.default
        elif normalized_param_type in dependency_cache:
            func_args[param_name] = dependency_cache[normalized_param_type]
        elif param_name in dependency_cache:
            func_args[param_name] = dependency_cache[param_name]
        else:
            name_cache = {
                get_dependency_name(_cache): _cache for _cache in dependency_cache
            }
            if (
                isinstance(normalized_param_type, str)
                and normalized_param_type in name_cache
            ):
                func_args[param_name] = dependency_cache[
                    name_cache[normalized_param_type]
                ]
            elif param_name in name_cache:
                func_args[param_name] = dependency_cache[name_cache[param_name]]
            else:
                raise TypeError(
                    f"Cannot resolve parameter '{param_name}' for dependency '{dependent.__name__}'"
                )

    return func_args


def _collect_class_annotations(dependent: type[Any]) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    for base in reversed(dependent.__mro__):
        if base is object:
            continue
        raw_annotations = get_annotations(base)
        for name, value in raw_annotations.items():
            annotations[name] = _resolve_class_annotation(base, value)
    return annotations


def _resolve_class_annotation(base: type[Any], value: Any) -> Any:
    if not isinstance(value, str):
        return _normalize_dependency_target(value)
    module = sys.modules.get(base.__module__)
    globalns = dict(getattr(module, "__dict__", {}))
    localns = dict(vars(base))
    try:
        holder = type(
            "_DependencyAnnotationHolder",
            (),
            {"__annotations__": {"value": value}},
        )
        return _normalize_dependency_target(
            get_type_hints(holder, globalns=globalns, localns=localns)["value"]
        )
    except (KeyError, NameError, SyntaxError, TypeError):
        return value


async def _execute_class(
    dependent: type[_T],
    stack: AsyncExitStack | None,
    dependency_cache: dict[Any, Any],
) -> Any:
    values: dict[str, Any] = {}
    ann = _collect_class_annotations(dependent)
    for name, sub_dependent in inspect.getmembers(
        dependent, lambda x: isinstance(x, InnerDepends)
    ):
        assert isinstance(sub_dependent, InnerDepends)
        if sub_dependent.dependency is None:
            dependent_ann = ann.get(name)
            if dependent_ann is None:
                raise TypeError(
                    f"can not resolve dependency for attribute '{name}' in {dependent}"
                )
            sub_dependent.dependency = _normalize_dependency_target(dependent_ann)
        values[name] = await solve_dependencies(
            cast("Dependency[_T]", sub_dependent.dependency),
            use_cache=sub_dependent.use_cache,
            stack=stack,
            dependency_cache=dependency_cache,
        )
    depend_obj = cast(
        "_T | AbstractAsyncContextManager[_T] | AbstractContextManager[_T]",
        dependent.__new__(dependent),
    )
    from serpsage.components.loads import ComponentRegistry
    from serpsage.components.metering import MeteringEmitterBase
    from serpsage.components.tracking import TrackingEmitterBase
    from serpsage.core.workunit import ClockBase, WorkUnit
    from serpsage.settings.models import AppSettings

    if isinstance(depend_obj, WorkUnit):
        try:
            settings = dependency_cache[AppSettings]
            clock = dependency_cache[ClockBase]
            tracker = dependency_cache[TrackingEmitterBase]
            meter = dependency_cache[MeteringEmitterBase]
            registry = dependency_cache[ComponentRegistry]
        except KeyError as exc:
            raise TypeError(
                f"{get_dependency_name(exc.args[0])} must be available before constructing a WorkUnit"
            ) from exc
        depend_obj._wu_bootstrap(
            settings=settings,
            clock=clock,
            tracker=tracker,
            meter=meter,
            components=registry,
        )
    for key, value in values.items():
        setattr(depend_obj, key, value)
    marker = object()
    previous_self = dependency_cache.get("self", marker)
    dependency_cache["self"] = depend_obj
    try:
        init_method = cast("Callable[..., Any]", type(depend_obj).__init__)
        if init_method is not object.__init__:
            init_args = await _resolve_callable_args(
                init_method,
                stack,
                dependency_cache,
            )
            if isinstance(depend_obj, WorkUnit):
                depend_obj._wu_bind_injected(
                    *values.values(),
                    *(value for name, value in init_args.items() if name != "self"),
                )
            if inspect.iscoroutinefunction(init_method):
                await init_method(**init_args)
            else:
                init_method(**init_args)
        elif isinstance(depend_obj, WorkUnit):
            depend_obj._wu_bind_injected(*values.values())
    finally:
        if previous_self is marker:
            dependency_cache.pop("self", None)
        else:
            dependency_cache["self"] = previous_self

    depend: Any
    if isinstance(depend_obj, WorkUnit):
        depend = depend_obj
    elif isinstance(depend_obj, AbstractAsyncContextManager):
        if stack is None:
            raise TypeError("stack cannot be None when entering an async context")
        async_cm = cast("AbstractAsyncContextManager[_T]", depend_obj)
        depend = cast(
            "_T",
            await stack.enter_async_context(async_cm),
        )
    elif isinstance(depend_obj, AbstractContextManager):
        if stack is None:
            raise TypeError("stack cannot be None when entering a sync context")
        sync_cm = cast("AbstractContextManager[_T]", depend_obj)
        depend = cast(
            "_T",
            await stack.enter_async_context(sync_ctx_manager_wrapper(sync_cm)),
        )
    else:
        depend = depend_obj

    return cast("_T", depend)


async def solve_dependencies(
    dependent: Dependency[_T],
    *,
    use_cache: bool = True,
    stack: AsyncExitStack | None = None,
    dependency_cache: dict[Any, Any],
) -> _T:
    if isinstance(dependent, InnerDepends):
        use_cache = dependent.use_cache
        if not dependent.dependency:
            raise TypeError("dependent cannot be None")
        dependent = dependent.dependency

    if not dependent:
        raise TypeError("dependent cannot be None")

    if use_cache and dependent in dependency_cache:
        return cast("_T", dependency_cache[dependent])

    if isinstance(dependent, type):
        # type of dependent is Type[T] (Class, not instance)
        depend = await _execute_class(dependent, stack, dependency_cache)
    elif inspect.isasyncgenfunction(dependent):
        # type of dependent is Callable[[], AsyncGenerator[T, None]]
        if stack is None:
            raise TypeError(
                "stack cannot be None when entering an async generator context"
            )
        cm = asynccontextmanager(dependent)()
        depend = cast("_T", await stack.enter_async_context(cm))
    elif inspect.isgeneratorfunction(dependent):
        # type of dependent is Callable[[], Generator[T, None, None]]
        if stack is None:
            raise TypeError("stack cannot be None when entering a generator context")
        cm = sync_ctx_manager_wrapper(contextmanager(dependent)())
        depend = cast("_T", await stack.enter_async_context(cm))
    elif inspect.iscoroutinefunction(dependent) or inspect.isfunction(dependent):
        # type of dependent is Callable[..., T] | Callable[..., Awaitable[T]]
        depend = await _execute_callable(dependent, stack, dependency_cache)
    elif inspect.ismethod(dependent):
        # type of dependent is a bound method (instance method)
        depend = await _execute_callable(dependent.__func__, stack, dependency_cache)
    elif isinstance(dependent, object) and callable(dependent):
        # type of dependent is an instance with __call__ method (Callable class instance)
        call_method = dependent.__call__  # type: ignore
        if inspect.iscoroutinefunction(call_method) or inspect.isfunction(call_method):
            depend = await _execute_callable(call_method, stack, dependency_cache)
        else:
            raise TypeError(
                f"__call__ method in {dependent.__class__.__name__} is not a valid function"
            )
    elif isinstance(dependent, str):
        name_cache = {
            get_dependency_name(_cache): _cache for _cache in dependency_cache
        }
        if dependent in name_cache:
            depend = dependency_cache[name_cache[dependent]]
        else:
            raise TypeError(f"Dependent token '{dependent}' is not available")
    else:
        raise TypeError(f"Dependent {dependent} is not a class, function, or generator")

    dependency_cache[dependent] = depend  # pylint: disable=possibly-used-before-assignment
    return cast("_T", depend)
