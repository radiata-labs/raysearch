from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeAlias, TypeVar, cast, overload

import anyio
from pydantic import BaseModel

from serpsage.core.workunit import WorkUnit
from serpsage.models.llm import (
    ChatDictResult,
    ChatModelResult,
    ChatResultBase,
    ChatTextResult,
)

TModel = TypeVar("TModel", bound=BaseModel)

RetryOn: TypeAlias = (
    type[Exception]
    | tuple[type[Exception], ...]
    | Exception
    | tuple[Exception, ...]
    | Callable[[Exception], bool]
)


class LLMClientBase(WorkUnit, ABC):
    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        timeout_s: float | None = None,
    ) -> ChatTextResult: ...

    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        timeout_s: float | None = None,
    ) -> ChatDictResult: ...

    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        timeout_s: float | None = None,
    ) -> ChatModelResult[TModel]: ...

    @abstractmethod
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResultBase:
        raise NotImplementedError

    @overload
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        timeout_s: float | None = None,
        retries: int = 0,
        retry_delay_s: float = 0.0,
        retry_on: RetryOn | None = None,
    ) -> ChatTextResult: ...

    @overload
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        timeout_s: float | None = None,
        retries: int = 0,
        retry_delay_s: float = 0.0,
        retry_on: RetryOn | None = None,
    ) -> ChatDictResult: ...

    @overload
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        timeout_s: float | None = None,
        retries: int = 0,
        retry_delay_s: float = 0.0,
        retry_on: RetryOn | None = None,
    ) -> ChatModelResult[TModel]: ...

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        timeout_s: float | None = None,
        retries: int = 0,
        retry_delay_s: float = 0.0,
        retry_on: RetryOn | None = None,
    ) -> ChatResultBase:
        attempts = max(1, int(retries) + 1)
        delay_s = max(0.0, float(retry_delay_s))
        retry_func = _normalize_retry_on(retry_on)

        for attempt_index in range(attempts):
            try:
                return await self._chat(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    timeout_s=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                if attempt_index >= attempts - 1:
                    raise
                if retry_func is not None and not retry_func(exc):
                    raise
                if delay_s > 0:
                    await anyio.sleep(delay_s)
        raise RuntimeError("llm retry loop exhausted without result")


def _normalize_retry_on(retry_on: RetryOn | None) -> Callable[[Exception], bool] | None:
    if retry_on is None:
        return None
    if isinstance(retry_on, type) and issubclass(retry_on, Exception):
        return lambda exc: isinstance(exc, retry_on)
    if isinstance(retry_on, Exception):
        return lambda exc: isinstance(exc, type(retry_on))
    if isinstance(retry_on, tuple):
        classes: list[type[Exception]] = []
        for item in retry_on:
            if isinstance(item, type) and issubclass(item, Exception):
                classes.append(item)
                continue
            if isinstance(item, Exception):
                classes.append(type(item))
                continue
            raise TypeError(
                "retry_on tuple entries must be Exception types or instances"
            )
        if not classes:
            return lambda _exc: False
        normalized = tuple(dict.fromkeys(classes))
        return lambda exc: isinstance(exc, normalized)
    if callable(retry_on):
        predicate = cast("Callable[[Exception], bool]", retry_on)
        return lambda exc: bool(predicate(exc))
    raise TypeError(
        "retry_on must be Exception type/instance, tuple of Exception values, "
        "callable, or None"
    )
