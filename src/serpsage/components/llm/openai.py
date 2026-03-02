from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar, overload
from typing_extensions import override

from openai import AsyncOpenAI
from pydantic import BaseModel

from serpsage.components.llm.base import LLMClientBase
from serpsage.models.llm import (
    ChatDictResult,
    ChatModelResult,
    ChatResultBase,
    ChatTextResult,
    LLMUsage,
)

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
    from openai.types.completion_usage import CompletionUsage
    from openai.types.shared_params.response_format_json_object import (
        ResponseFormatJSONObject,
    )
    from openai.types.shared_params.response_format_json_schema import (
        ResponseFormatJSONSchema,
    )

    from serpsage.components.http.base import HttpClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import LLMModelSettings
TModel = TypeVar("TModel", bound=BaseModel)


class OpenAIClient(LLMClientBase):
    _SCHEMA_NAME = "SerpSageOverview"

    def __init__(
        self, *, rt: Runtime, http: HttpClientBase, model_cfg: LLMModelSettings
    ) -> None:
        super().__init__(rt=rt)
        self.bind_deps(http)
        self._model_cfg = model_cfg
        llm = self._model_cfg
        self.client = AsyncOpenAI(
            api_key=llm.api_key,
            base_url=llm.base_url,
            timeout=float(llm.timeout_s),
            max_retries=int(llm.max_retries),
            default_headers=dict(llm.headers or {}),
            http_client=http.client,
        )

    @overload
    async def _chat(
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
    async def _chat(
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
    async def _chat(
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
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatResultBase:
        llm = self._model_cfg
        if not llm.api_key:
            raise RuntimeError("missing LLM api_key")
        response_schema, response_model = self.resolve_response_format(
            response_format,
            format_override=format_override,
        )
        if (
            response_model is not None
            and llm.enable_structured
            and format_override is None
        ):
            pydantic_completion = await self._create_pydantic_completion(
                model=model,
                messages=self._to_openai_messages(messages),
                temperature=float(llm.temperature),
                timeout=float(timeout_s or llm.timeout_s),
                response_format=response_model,
                **kwargs,
            )
            text = self._extract_text(pydantic_completion)
            parsed = self._extract_pydantic_text(pydantic_completion)
            usage = self._to_usage(getattr(pydantic_completion, "usage", None))
            if parsed is None:
                parsed = response_model.model_validate_json(text)
            return ChatModelResult(
                text=text,
                data=parsed,
                usage=usage,
            )
        completion = await self._create_completion(
            model=model,
            messages=self._to_openai_messages(
                messages, response_model, llm.enable_structured
            ),
            temperature=float(llm.temperature),
            timeout=float(timeout_s or llm.timeout_s),
            response_format=self._build_response_format_payload(
                schema=response_schema,
                enable_structured=bool(llm.enable_structured),
            ),
            **kwargs,
        )
        text = self._extract_text(completion)
        usage = self._to_usage(getattr(completion, "usage", None))
        if response_schema is None:
            return ChatTextResult(text=text, usage=usage)
        if response_model is not None:
            return ChatModelResult(
                text=text,
                data=response_model.model_validate_json(text),
                usage=usage,
            )
        data = self._parse_json_object(text)
        return ChatDictResult(text=text, data=data, usage=usage)

    async def _create_pydantic_completion(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        timeout: float,
        response_format: type[TModel],
        **kwargs: Any,
    ) -> ParsedChatCompletion[TModel]:
        return await self.client.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format,
            **kwargs,
        )

    async def _create_completion(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        timeout: float,
        response_format: ResponseFormatJSONSchema | ResponseFormatJSONObject | None,
        **kwargs: Any,
    ) -> ChatCompletion:
        if response_format is None:
            completion: ChatCompletion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )
        else:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                response_format=response_format,
                **kwargs,
            )
        return completion

    @classmethod
    def _build_response_format_payload(
        cls,
        *,
        schema: dict[str, object] | None,
        enable_structured: bool,
    ) -> ResponseFormatJSONSchema | ResponseFormatJSONObject | None:
        if schema is None:
            return None
        if enable_structured:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": cls._SCHEMA_NAME,
                    "schema": schema,
                    "strict": True,
                },
            }
        return {"type": "json_object"}

    def _to_openai_messages(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel] | None = None,
        enable_structured: bool = True,
    ) -> list[ChatCompletionMessageParam]:
        out: list[ChatCompletionMessageParam] = []
        if not enable_structured and response_model is not None:
            structure_prompt = self.get_format_instructions(response_model)
            if structure_prompt:
                out.append({"role": "system", "content": structure_prompt})
        for msg in messages:
            role = str(msg.get("role") or "user").strip().lower()
            content = str(msg.get("content") or "")
            if role == "system":
                out.append({"role": "system", "content": content})
                continue
            if role == "assistant":
                out.append({"role": "assistant", "content": content})
                continue
            out.append({"role": "user", "content": content})
        if not out:
            out.append({"role": "user", "content": ""})
        return out

    @staticmethod
    def _extract_pydantic_text(resp: ParsedChatCompletion[TModel]) -> TModel | None:
        choices = list(getattr(resp, "choices", []) or [])
        if not choices:
            return None
        head = choices[0]
        message = getattr(head, "message", None)
        return getattr(message, "parsed", None)

    @staticmethod
    def _extract_text(resp: ChatCompletion | ParsedChatCompletion[TModel]) -> str:
        choices = list(getattr(resp, "choices", []) or [])
        if not choices:
            return ""
        head = choices[0]
        message = getattr(head, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        return str(content or "")

    @staticmethod
    def _to_usage(usage: CompletionUsage | None) -> LLMUsage:
        if usage is None:
            return LLMUsage()
        return LLMUsage(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, object]:
        payload: object
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if 0 <= start < end:
                payload = json.loads(content[start : end + 1])
            else:
                raise
        if not isinstance(payload, dict):
            raise TypeError("structured LLM response must be a JSON object")
        return {str(k): v for k, v in payload.items()}


__all__ = ["OpenAIClient"]
