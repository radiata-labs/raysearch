from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast, overload
from typing_extensions import TypeVar, override

from pydantic import BaseModel, ValidationError

from serpsage.components.http.base import HttpClientBase
from serpsage.components.llm.base import LLMClientBase, LLMConfig, LLMModelConfig
from serpsage.dependencies import Depends
from serpsage.models.components.llm import (
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

TModel = TypeVar("TModel", bound=BaseModel)


class OpenAIModelConfig(LLMConfig):
    __setting_family__ = "llm"
    __setting_name__ = "openai"

    @classmethod
    @override
    def inject_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> dict[str, Any]:
        return cls.inject_model_env(
            raw,
            env=env,
            api_key_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
        )


class OpenAIClient(LLMClientBase[OpenAIModelConfig]):
    _SCHEMA_NAME = "SerpSageOverview"

    http: HttpClientBase = Depends()

    def __init__(
        self,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required for OpenAIClient") from exc
        self._async_openai = AsyncOpenAI

    def _build_client(self, llm: LLMModelConfig) -> Any:
        return self._async_openai(
            api_key=llm.api_key,
            base_url=llm.base_url,
            timeout=float(llm.timeout_s),
            max_retries=int(llm.max_retries),
            default_headers=dict(llm.headers or {}),
            http_client=self.http.client,
        )

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
        llm = self.resolve_model_config(model, route_name=self.pop_route_name(kwargs))
        if not llm.api_key:
            raise RuntimeError("missing LLM api_key")
        client = self._build_client(llm)
        provider_model = str(llm.model)
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
                client=client,
                model=provider_model,
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
                parsed = self._validate_model_response(
                    response_model=response_model,
                    content=text,
                )
            return ChatModelResult(text=text, data=parsed, usage=usage)
        completion = await self._create_completion(
            client=client,
            model=provider_model,
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
                data=self._validate_model_response(
                    response_model=response_model,
                    content=text,
                ),
                usage=usage,
            )
        data = self._parse_json_object(text)
        return ChatDictResult(text=text, data=data, usage=usage)

    async def _create_pydantic_completion(
        self,
        *,
        client: Any,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        timeout: float,
        response_format: type[TModel],
        **kwargs: Any,
    ) -> ParsedChatCompletion[TModel]:
        return cast(
            "ParsedChatCompletion[TModel]",
            await client.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                response_format=response_format,
                **kwargs,
            ),
        )

    async def _create_completion(
        self,
        *,
        client: Any,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        timeout: float,
        response_format: ResponseFormatJSONSchema | ResponseFormatJSONObject | None,
        **kwargs: Any,
    ) -> ChatCompletion:
        if response_format is None:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )
            return cast("ChatCompletion", response)
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format,
            **kwargs,
        )
        return cast("ChatCompletion", response)

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
            elif role == "assistant":
                out.append({"role": "assistant", "content": content})
            else:
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

    @classmethod
    def _validate_model_response(
        cls,
        *,
        response_model: type[TModel],
        content: str,
    ) -> TModel:
        last_error: ValidationError | None = None
        for candidate in cls._iter_json_candidates(content):
            try:
                return response_model.model_validate_json(candidate)
            except ValidationError as exc:
                last_error = exc
        payload = cls._parse_json_object(content)
        try:
            return response_model.model_validate(payload)
        except ValidationError:
            if last_error is not None:
                raise last_error from None
            raise

    @classmethod
    def _parse_json_object(cls, content: str) -> dict[str, object]:
        last_error: json.JSONDecodeError | None = None
        payload: object = None
        for candidate in cls._iter_json_candidates(content):
            try:
                payload = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        else:
            if last_error is not None:
                raise last_error
            raise json.JSONDecodeError(
                "structured LLM response is not valid JSON", "", 0
            )
        if not isinstance(payload, dict):
            raise TypeError("structured LLM response must be a JSON object")
        return {str(k): v for k, v in payload.items()}

    @classmethod
    def _iter_json_candidates(cls, content: str) -> tuple[str, ...]:
        candidates: list[str] = []

        def add(candidate: str) -> None:
            token = str(candidate)
            if not token or token in candidates:
                return
            candidates.append(token)

        add(content)
        sanitized = cls._sanitize_json_text(content)
        add(sanitized)

        for candidate in candidates:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if 0 <= start < end:
                add(candidate[start : end + 1])

        return tuple(candidates)

    @staticmethod
    def _sanitize_json_text(content: str) -> str:
        if not content:
            return ""
        out: list[str] = []
        in_string = False
        escaped = False
        changed = False
        for char in content:
            if in_string:
                if escaped:
                    out.append(char)
                    escaped = False
                    continue
                if char == "\\":
                    out.append(char)
                    escaped = True
                    continue
                if char == '"':
                    out.append(char)
                    in_string = False
                    continue
                if char == "\n":
                    out.append("\\n")
                    changed = True
                    continue
                if char == "\r":
                    out.append("\\r")
                    changed = True
                    continue
                if char == "\t":
                    out.append("\\t")
                    changed = True
                    continue
                codepoint = ord(char)
                if codepoint < 0x20:
                    out.append(f"\\u{codepoint:04x}")
                    changed = True
                    continue
                out.append(char)
                continue
            out.append(char)
            if char == '"':
                in_string = True
        if not changed:
            return content
        return "".join(out)


__all__ = ["OpenAIClient", "OpenAIModelConfig"]
