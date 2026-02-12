from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from serpsage.settings.load import load_settings
from serpsage.settings.models import AppSettings


def _overview_config() -> dict:
    return {
        "enabled": True,
        "use_model": "gpt-4.1-mini",
        "models": [
            {
                "name": "gpt-4.1-mini",
                "backend": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-openai",
                "model": "gpt-4.1-mini",
                "timeout_s": 60.0,
                "max_retries": 2,
                "temperature": 0.0,
                "headers": {},
                "schema_strict": True,
            },
            {
                "name": "gemini-3-flash",
                "backend": "gemini",
                "base_url": "https://generativelanguage.googleapis.com/v1",
                "api_key": "sk-gemini",
                "model": "gemini-3-flash",
                "timeout_s": 60.0,
                "max_retries": 2,
                "temperature": 0.0,
                "headers": {},
                "schema_strict": True,
            },
        ],
        "max_sources": 8,
        "max_chunks_per_source": 2,
        "max_chunk_chars": 900,
        "max_output_tokens": 600,
        "max_prompt_chars": 32000,
        "cache_ttl_s": 0,
        "self_heal_retries": 1,
        "force_language": "auto",
    }


def test_overview_new_config_resolve_model() -> None:
    cfg = _overview_config()
    cfg["use_model"] = "gemini-3-flash"
    settings = AppSettings.model_validate({"overview": cfg})
    active = settings.overview.resolve_model()
    assert active.name == "gemini-3-flash"
    assert active.backend == "gemini"
    assert active.model == "gemini-3-flash"


def test_overview_models_must_not_be_empty() -> None:
    cfg = _overview_config()
    cfg["models"] = []
    with pytest.raises(ValidationError, match="overview.models must contain at least one model"):
        AppSettings.model_validate({"overview": cfg})


def test_overview_use_model_must_exist() -> None:
    cfg = _overview_config()
    cfg["use_model"] = "missing-model"
    with pytest.raises(
        ValidationError,
        match="overview.use_model must match one of overview.models\\[\\]\\.name",
    ):
        AppSettings.model_validate({"overview": cfg})


def test_overview_model_name_must_be_unique() -> None:
    cfg = _overview_config()
    cfg["models"][1]["name"] = "gpt-4.1-mini"
    with pytest.raises(ValidationError, match="duplicate overview model name"):
        AppSettings.model_validate({"overview": cfg})


def test_overview_legacy_keys_are_forbidden() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AppSettings.model_validate(
            {
                "overview": {
                    "enabled": True,
                    "backend": "openai",
                    "openai": {"llm": {"api_key": "sk-legacy"}},
                    "null": {},
                }
            }
        )


def test_example_config_loads_with_new_overview_shape() -> None:
    settings = load_settings("src/search_config_example.yaml")
    active = settings.overview.resolve_model()
    assert active.name == settings.overview.use_model
    assert active.model == "gpt-4.1-mini"


def _write_config(tmp_path: Path, payload: dict) -> str:
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_overview_env_fallback_by_backend(tmp_path) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "overview": {
                "enabled": True,
                "use_model": "openai-model",
                "models": [
                    {
                        "name": "openai-model",
                        "backend": "openai",
                        "api_key": None,
                        "base_url": None,
                    },
                    {
                        "name": "gemini-model",
                        "backend": "gemini",
                        "api_key": "",
                        "base_url": "",
                    },
                ],
            }
        },
    )
    settings = load_settings(
        cfg_path,
        env={
            "OPENAI_API_KEY": "env-openai-key",
            "OPENAI_BASE_URL": "https://env.openai.test/v1",
            "GEMINI_API_KEY": "env-gemini-key",
            "GEMINI_BASE_URL": "https://env.gemini.test/v1",
        },
    )
    m0 = settings.overview.models[0]
    m1 = settings.overview.models[1]
    assert m0.api_key == "env-openai-key"
    assert m0.base_url == "https://env.openai.test/v1"
    assert m1.api_key == "env-gemini-key"
    assert m1.base_url == "https://env.gemini.test/v1"


def test_overview_yaml_non_empty_overrides_env(tmp_path) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "overview": {
                "enabled": True,
                "use_model": "openai-model",
                "models": [
                    {
                        "name": "openai-model",
                        "backend": "openai",
                        "api_key": "yaml-openai-key",
                        "base_url": "https://yaml.openai.test/v1",
                    },
                    {
                        "name": "gemini-model",
                        "backend": "gemini",
                        "api_key": "yaml-gemini-key",
                        "base_url": "https://yaml.gemini.test/v1",
                    },
                ],
            }
        },
    )
    settings = load_settings(
        cfg_path,
        env={
            "OPENAI_API_KEY": "env-openai-key",
            "OPENAI_BASE_URL": "https://env.openai.test/v1",
            "GEMINI_API_KEY": "env-gemini-key",
            "GEMINI_BASE_URL": "https://env.gemini.test/v1",
        },
    )
    m0 = settings.overview.models[0]
    m1 = settings.overview.models[1]
    assert m0.api_key == "yaml-openai-key"
    assert m0.base_url == "https://yaml.openai.test/v1"
    assert m1.api_key == "yaml-gemini-key"
    assert m1.base_url == "https://yaml.gemini.test/v1"


def test_overview_missing_yaml_fields_fallback_to_env(tmp_path) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "overview": {
                "enabled": True,
                "use_model": "gemini-model",
                "models": [
                    {
                        "name": "gemini-model",
                        "backend": "gemini",
                    }
                ],
            }
        },
    )
    settings = load_settings(
        cfg_path,
        env={
            "GEMINI_API_KEY": "env-gemini-key",
            "GEMINI_BASE_URL": "https://env.gemini.test/v1",
        },
    )
    model = settings.overview.models[0]
    assert model.api_key == "env-gemini-key"
    assert model.base_url == "https://env.gemini.test/v1"


def test_searxng_env_override_regression(tmp_path) -> None:
    cfg_path = _write_config(
        tmp_path,
        {
            "provider": {
                "backend": "searxng",
                "searxng": {
                    "base_url": "https://yaml.searxng.test/search",
                    "api_key": "yaml-search-key",
                },
            }
        },
    )
    settings = load_settings(
        cfg_path,
        env={
            "SEARXNG_BASE_URL": "https://env.searxng.test/search",
            "SEARCH_API_KEY": "env-search-key",
        },
    )
    assert settings.provider.searxng.base_url == "https://env.searxng.test/search"
    assert settings.provider.searxng.api_key == "env-search-key"
