from __future__ import annotations

import httpx
import pytest

import serpsage.overview.openai as mod
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import CoreRuntime
from serpsage.models.llm import ChatJSONResult
from serpsage.overview.openai import OpenAIClient
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class _DummyAsyncOpenAI:
    def __init__(self, **kwargs):  # noqa: ANN003
        self._ctor_kwargs = dict(kwargs)
        self._create_kwargs = None

        class _Chat:
            def __init__(self, outer):  # noqa: ANN001
                self.completions = self
                self._outer = outer

            async def create(self, **kwargs):  # noqa: ANN003
                self._outer._create_kwargs = dict(kwargs)

                class _Msg:
                    content = '{"summary":"ok","key_points":["p1"],"citations":[]}'

                class _Choice:
                    message = _Msg()

                class _Usage:
                    prompt_tokens = 1
                    completion_tokens = 2
                    total_tokens = 3

                class _Resp:
                    choices = [_Choice()]
                    usage = _Usage()

                return _Resp()

        self.chat = _Chat(self)


class _FakeClock(ClockBase):
    def now_ms(self) -> int:
        return 0


@pytest.mark.anyio
async def test_official_client_passes_response_format_and_limits(monkeypatch):
    monkeypatch.setattr(mod, "AsyncOpenAI", _DummyAsyncOpenAI)

    settings = AppSettings.model_validate(
        {"overview": {"enabled": True, "llm": {"api_key": "dummy"}}}
    )
    rt = CoreRuntime(settings=settings, telemetry=NoopTelemetry(), clock=_FakeClock())

    async with httpx.AsyncClient() as http:
        client = OpenAIClient(rt=rt, http=http)
        res = await client.chat_json(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            schema={"type": "object"},
            timeout_s=1.0,
        )

    assert isinstance(res, ChatJSONResult)
    assert res.data["summary"] == "ok"
    assert res.usage.prompt_tokens == 1
    assert res.usage.completion_tokens == 2
    assert res.usage.total_tokens == 3
    dummy = client.client  # type: ignore[attr-defined]
    assert dummy._create_kwargs is not None
    assert "max_completion_tokens" not in dummy._create_kwargs
    assert "max_tokens" not in dummy._create_kwargs
    assert "response_format" in dummy._create_kwargs
    assert dummy._create_kwargs["response_format"]["type"] in {
        "json_schema",
        "json_object",
    }
