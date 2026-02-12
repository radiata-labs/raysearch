from __future__ import annotations

import types

import anyio
import pytest

pytest.importorskip("google.genai")
from google.genai import errors

from serpsage.app.bootstrap import build_runtime
from serpsage.components.overview.gemini import GeminiClient
from serpsage.core.runtime import Overrides
from serpsage.settings.models import AppSettings


def _build_client(*, schema_strict: bool) -> GeminiClient:
    settings = AppSettings.model_validate(
        {
            "overview": {
                "enabled": True,
                "use_model": "g1",
                "models": [
                    {
                        "name": "g1",
                        "backend": "gemini",
                        "base_url": "https://generativelanguage.googleapis.com/v1",
                        "api_key": "sk-gemini",
                        "model": "gemini-3-flash",
                        "timeout_s": 60.0,
                        "max_retries": 2,
                        "temperature": 0.0,
                        "headers": {},
                        "schema_strict": schema_strict,
                    }
                ],
            }
        }
    )
    rt = build_runtime(settings=settings, overrides=Overrides())
    model_cfg = settings.overview.resolve_model()
    return GeminiClient(rt=rt, model_cfg=model_cfg)


def test_gemini_chat_json_uses_parsed_and_maps_usage() -> None:
    client = _build_client(schema_strict=True)
    captured: dict[str, object] = {}

    async def fake_generate_content(*, model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return types.SimpleNamespace(
            parsed={"summary": "ok"},
            text='{"summary":"fallback"}',
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=11,
                candidates_token_count=7,
                total_token_count=18,
            ),
        )

    client.client = types.SimpleNamespace(
        aio=types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=fake_generate_content)
        )
    )

    async def run():
        return await client.chat_json(
            model="gemini-3-flash",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            schema={"type": "object"},
            timeout_s=12.0,
        )

    out = anyio.run(run)
    cfg = captured["config"]
    assert out.data == {"summary": "ok"}
    assert out.usage.prompt_tokens == 11
    assert out.usage.completion_tokens == 7
    assert out.usage.total_tokens == 18
    assert getattr(cfg, "response_mime_type") == "application/json"
    assert getattr(cfg, "response_json_schema") == {"type": "object"}


def test_gemini_chat_json_non_strict_without_schema() -> None:
    client = _build_client(schema_strict=False)
    captured: dict[str, object] = {}

    async def fake_generate_content(*, model, contents, config):
        captured["config"] = config
        return types.SimpleNamespace(
            parsed=None,
            text='{"summary":"from_text"}',
            usage_metadata=None,
        )

    client.client = types.SimpleNamespace(
        aio=types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=fake_generate_content)
        )
    )

    async def run():
        return await client.chat_json(
            model="gemini-3-flash",
            messages=[{"role": "user", "content": "hello"}],
            schema={"type": "object"},
            timeout_s=None,
        )

    out = anyio.run(run)
    cfg = captured["config"]
    assert out.data == {"summary": "from_text"}
    assert out.usage.total_tokens == 0
    assert getattr(cfg, "response_mime_type") == "application/json"
    assert getattr(cfg, "response_json_schema") is None


def test_gemini_chat_json_schema_rejected_fallback_once() -> None:
    client = _build_client(schema_strict=True)
    seen_configs: list[object] = []
    calls = {"n": 0}

    async def fake_generate_content(*, model, contents, config):
        calls["n"] += 1
        seen_configs.append(config)
        if calls["n"] == 1:
            raise errors.ClientError(400, {"error": {"message": "invalid schema"}})
        return types.SimpleNamespace(
            parsed=None,
            text="head {\"ok\":true} tail",
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=1,
                candidates_token_count=1,
                total_token_count=2,
            ),
        )

    client.client = types.SimpleNamespace(
        aio=types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=fake_generate_content)
        )
    )

    async def run():
        return await client.chat_json(
            model="gemini-3-flash",
            messages=[{"role": "user", "content": "hello"}],
            schema={"type": "object"},
            timeout_s=None,
        )

    out = anyio.run(run)
    assert calls["n"] == 2
    assert out.data == {"ok": True}
    assert getattr(seen_configs[0], "response_json_schema") == {"type": "object"}
    assert getattr(seen_configs[1], "response_json_schema") is None
