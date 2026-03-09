from __future__ import annotations

from typing import Any, NoReturn, TypeVar, overload
from typing_extensions import override

from pydantic import BaseModel

from serpsage.components.base import ComponentMeta
from serpsage.components.llm.base import LLMClientBase, LLMRouterConfig
from serpsage.components.registry import register_component
from serpsage.models.components.llm import (
    ChatDictResult,
    ChatModelResult,
    ChatResultBase,
    ChatTextResult,
)

TModel = TypeVar("TModel", bound=BaseModel)
_NULL_LLM_META = ComponentMeta(
    family="llm",
    name="null",
    version="1.0.0",
    summary="Null LLM client.",
    provides=("llm.client",),
    config_model=LLMRouterConfig,
)


@register_component(meta=_NULL_LLM_META)
class NullLLMClient(LLMClientBase[LLMRouterConfig]):
    meta = _NULL_LLM_META
    _NOT_CONFIGURED_MESSAGE = "LLM is not configured (missing api_key or disabled)."

    def __init__(
        self,
        *,
        rt: object,
        config: LLMRouterConfig,
    ) -> None:
        super().__init__(rt=rt, config=config)

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
    @override
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
        _ = model, messages, response_format, format_override, timeout_s, kwargs
        self._raise_not_configured()

    @classmethod
    def _raise_not_configured(cls) -> NoReturn:
        raise RuntimeError(cls._NOT_CONFIGURED_MESSAGE)


__all__ = ["NullLLMClient"]
