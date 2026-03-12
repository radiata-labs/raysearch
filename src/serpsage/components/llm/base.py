from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeAlias, cast, overload
from typing_extensions import TypeVar

import anyio
from pydantic import BaseModel, Field

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.llm import (
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


class LLMModelConfig(ComponentConfigBase):
    name: str = "gpt-4.1-mini"
    base_url: str | None = None
    api_key: str | None = None
    model: str = "gpt-4.1-mini"
    timeout_s: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    headers: dict[str, str] = Field(default_factory=dict)
    enable_structured: bool = True


class LLMConfig(ComponentConfigBase):
    models: list[LLMModelConfig] = Field(default_factory=list)

    @classmethod
    def normalize_models_payload(cls, raw: dict[str, Any]) -> dict[str, Any]:
        payload = dict(raw)
        raw_models = payload.get("models")
        if isinstance(raw_models, list):
            payload["models"] = [
                cls._normalize_model_payload(item) for item in raw_models
            ]
            return payload
        legacy_model = cls._extract_legacy_model_payload(payload)
        if legacy_model is not None:
            payload["models"] = [legacy_model]
        return payload

    @classmethod
    def inject_model_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
        api_key_env: str = "",
        base_url_env: str = "",
    ) -> dict[str, Any]:
        payload = cls.normalize_models_payload(raw)
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            return payload
        models: list[dict[str, Any]] = []
        for item in raw_models:
            model_payload = cls._normalize_model_payload(item)
            if (
                api_key_env
                and not model_payload.get("api_key")
                and env.get(api_key_env)
            ):
                model_payload["api_key"] = env[api_key_env]
            if (
                base_url_env
                and not model_payload.get("base_url")
                and env.get(base_url_env)
            ):
                model_payload["base_url"] = env[base_url_env]
            models.append(model_payload)
        payload["models"] = models
        return payload

    @staticmethod
    def _normalize_model_payload(item: object) -> dict[str, Any]:
        if isinstance(item, LLMModelConfig):
            return item.model_dump(mode="python")
        if isinstance(item, BaseModel):
            return item.model_dump(mode="python")
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError("llm models entries must be mappings")

    @classmethod
    def _extract_legacy_model_payload(
        cls,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        legacy_fields = (
            "name",
            "base_url",
            "api_key",
            "model",
            "timeout_s",
            "max_retries",
            "temperature",
            "headers",
            "enable_structured",
        )
        if not any(field in payload for field in legacy_fields):
            return None
        model_payload: dict[str, Any] = {}
        for field in legacy_fields:
            if field not in payload:
                continue
            model_payload[field] = payload.pop(field)
        return model_payload


LLMConfigT = TypeVar(
    "LLMConfigT",
    bound=ComponentConfigBase,
    default=ComponentConfigBase,
)


class LLMClientBase(ComponentBase[LLMConfigT], ABC, Generic[LLMConfigT]):
    __di_contract__ = True

    def configured_models(self) -> tuple[LLMModelConfig, ...]:
        models = getattr(self.config, "models", ()) or ()
        return tuple(model for model in models if isinstance(model, LLMModelConfig))

    def resolve_model_config(
        self,
        model: str,
        *,
        route_name: str | None = None,
    ) -> LLMModelConfig:
        configured = self.configured_models()
        if not configured:
            raise RuntimeError(
                f"{type(self).__name__} requires at least one configured llm model"
            )
        normalized_route = str(route_name or "").strip()
        if normalized_route:
            for item in configured:
                if str(item.name).strip() == normalized_route:
                    return item
        normalized_model = str(model or "").strip()
        if normalized_model:
            for item in configured:
                if normalized_model in {
                    str(item.name).strip(),
                    str(item.model).strip(),
                }:
                    return item
        if len(configured) == 1:
            return configured[0]
        available = ", ".join(
            f"{str(item.name).strip()}->{str(item.model).strip()}"
            for item in configured
        )
        raise RuntimeError(
            f"llm model `{normalized_route or normalized_model}` is not configured "
            f"for `{type(self).__name__}`; available routes: {available}"
        )

    @staticmethod
    def pop_route_name(kwargs: dict[str, Any]) -> str | None:
        route_name = kwargs.pop("_route_name", None)
        if route_name is None:
            return None
        normalized = str(route_name).strip()
        return normalized or None

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
        schema = dict(pydantic_object.model_json_schema().items())
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)
        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)


_PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.
As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.
Here is the output schema:
```
{schema}
```"""


__all__ = [
    "LLMClientBase",
    "LLMConfig",
    "LLMModelConfig",
    "RetryOn",
]
