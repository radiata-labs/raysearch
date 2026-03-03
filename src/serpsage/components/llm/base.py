from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar, cast, overload

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
RetryPredicate = Callable[[Exception], bool]
RetryOn: TypeAlias = (
    type[Exception]
    | tuple[type[Exception], ...]
    | Exception
    | tuple[Exception, ...]
    | RetryPredicate
)


class LLMClientBase(WorkUnit, ABC):
    @overload
    async def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        format_override: None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatTextResult: ...
    @overload
    async def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        format_override: None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatDictResult: ...
    # `format_override` is only meaningful for BaseModel response_format.
    @overload
    async def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatModelResult[TModel]: ...
    @abstractmethod
    async def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatResultBase:
        """Provider chat implementation.
        `format_override` is only valid when `response_format` is `type[BaseModel]`.
        Providers must use this override schema for constrained output and still
        parse the response into that BaseModel.
        """
        raise NotImplementedError

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        format_override: None = None,
        timeout_s: float | None = None,
        retries: int = 0,
        retry_delay_s: float = 0.0,
        retry_on: RetryOn | None = None,
        **kwargs: Any,
    ) -> ChatTextResult: ...
    @overload
    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        format_override: None = None,
        timeout_s: float | None = None,
        retries: int = 0,
        retry_delay_s: float = 0.0,
        retry_on: RetryOn | None = None,
        **kwargs: Any,
    ) -> ChatDictResult: ...
    # `format_override` can be used only when response_format is BaseModel type.
    @overload
    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        retries: int = 0,
        retry_delay_s: float = 0.0,
        retry_on: RetryOn | None = None,
        **kwargs: Any,
    ) -> ChatModelResult[TModel]: ...
    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        retries: int = 0,
        retry_delay_s: float = 0.0,
        retry_on: RetryOn | None = None,
        **kwargs: Any,
    ) -> ChatResultBase:
        """Public chat entrypoint with retries.
        `format_override` is only allowed when `response_format` is
        `type[BaseModel]`. When provided, the override schema is used instead of
        `BaseModel.model_json_schema()`, but output is still validated into the
        same BaseModel type.
        """
        self._validate_format_override(
            response_format=response_format,
            format_override=format_override,
        )
        attempts = max(1, int(retries) + 1)
        delay_s = max(0.0, float(retry_delay_s))
        retry_func = self._normalize_retry_on(retry_on)
        for attempt_index in range(attempts):
            try:
                if response_format is None:
                    return await self._create(
                        model=model,
                        messages=messages,
                        response_format=None,
                        format_override=None,
                        timeout_s=timeout_s,
                        **kwargs,
                    )
                if isinstance(response_format, dict):
                    return await self._create(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        format_override=None,
                        timeout_s=timeout_s,
                        **kwargs,
                    )
                if isinstance(response_format, type) and issubclass(
                    response_format, BaseModel
                ):
                    return await self._create(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        format_override=format_override,
                        timeout_s=timeout_s,
                        **kwargs,
                    )
                raise TypeError(
                    "response_format must be dict[str, object] | type[BaseModel] | None"
                )
            except Exception as exc:  # noqa: BLE001
                if attempt_index >= attempts - 1:
                    raise
                if retry_func is not None and not retry_func(exc):
                    raise
                if delay_s > 0:
                    await anyio.sleep(delay_s)
        raise RuntimeError("llm retry loop exhausted without result")

    @staticmethod
    def _normalize_retry_on(retry_on: RetryOn | None) -> RetryPredicate | None:
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
            predicate = cast("RetryPredicate", retry_on)
            return lambda exc: bool(predicate(exc))
        raise TypeError(
            "retry_on must be Exception type/instance, tuple of Exception values, "
            "callable, or None"
        )

    @staticmethod
    def resolve_response_format(
        response_format: dict[str, object] | type[BaseModel] | None,
        format_override: dict[str, object] | None = None,
    ) -> tuple[dict[str, object] | None, type[BaseModel] | None]:
        LLMClientBase._validate_format_override(
            response_format=response_format,
            format_override=format_override,
        )
        if response_format is None:
            return None, None
        if isinstance(response_format, dict):
            return dict(response_format), None
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            if format_override is not None:
                return dict(format_override), response_format
            return response_format.model_json_schema(), response_format
        raise TypeError(
            "response_format must be dict[str, object] | type[BaseModel] | None"
        )

    @staticmethod
    def _validate_format_override(
        *,
        response_format: dict[str, object] | type[BaseModel] | None,
        format_override: dict[str, object] | None,
    ) -> None:
        if format_override is None:
            return
        if not (
            isinstance(response_format, type) and issubclass(response_format, BaseModel)
        ):
            raise TypeError(
                "format_override is only allowed when response_format is "
                "type[BaseModel]"
            )

    @staticmethod
    def get_format_instructions(pydantic_object: type[BaseModel]) -> str:
        """Return the format instructions for the JSON output.
        Returns:
            The format instructions for the JSON output.
        """
        # Copy schema to avoid altering original Pydantic schema.
        schema = dict(pydantic_object.model_json_schema().items())
        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)
        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)


_PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.
As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.
Here is the output schema:
```
{schema}
```"""
