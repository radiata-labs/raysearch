from __future__ import annotations

import httpx
import pytest

import serpsage.overview.openai as mod
from serpsage.app.runtime import CoreRuntime
from serpsage.contracts.llm import ChatJSONResult
from serpsage.overview.openai import OpenAIClient
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class _FakeClock:
    def now_ms(self) -> int:
        return 0


class _BadSchemaError(Exception):
    def __str__(self) -> str:
        return (
            "Error code: 400 - {'error': {'message': \"Invalid schema for response_format "
            "'SerpSageOverview': In context=(), 'additionalProperties' is required "
            "to be supplied and to be false.\", 'type': 'invalid_request_error', "
            "'param': 'response_format'}}"
        )


class _DummyAsyncOpenAI:
    def __init__(self, **kwargs):  # noqa: ANN003
        self._ctor_kwargs = dict(kwargs)
        self._calls = 0

        class _Chat:
            def __init__(self, outer):  # noqa: ANN001
                self.completions = self
                self._outer = outer

            async def create(self, **kwargs):  # noqa: ANN003, ARG002
                self._outer._calls += 1
                # First call fails (strict schema), second succeeds (fallback).
                if self._outer._calls == 1:
                    raise _BadSchemaError

                class _Msg:
                    content = '{"summary":"ok","key_points":["p1"],"citations":[]}'

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]
                    usage = None

                return _Resp()

        self.chat = _Chat(self)


@pytest.mark.anyio
async def test_schema_rejected_falls_back_to_json_object(monkeypatch):
    monkeypatch.setattr(mod, "AsyncOpenAI", _DummyAsyncOpenAI)

    settings = AppSettings.model_validate(
        {
            "overview": {
                "enabled": True,
                "schema_strict": True,
                "llm": {"api_key": "dummy"},
            }
        }
    )
    rt = CoreRuntime(settings=settings, telemetry=NoopTelemetry(), clock=_FakeClock())

    async with httpx.AsyncClient() as http:
        client = OpenAIClient(rt=rt, http=http)
        res = await client.chat_json(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            schema={"type": "object", "properties": {}},
            timeout_s=1.0,
        )

    assert isinstance(res, ChatJSONResult)
    assert res.data["summary"] == "ok"
