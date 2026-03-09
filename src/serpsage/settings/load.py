from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from serpsage.settings.models import AppSettings


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _llm_env_keys(backend: str) -> tuple[str | None, str | None]:
    key = str(backend or "").strip().lower()
    if key == "openai":
        return "OPENAI_API_KEY", "OPENAI_BASE_URL"
    if key == "gemini":
        return "GEMINI_API_KEY", "GEMINI_BASE_URL"
    return None, None


def _raw_llm_models(data: dict[str, Any]) -> list[dict[str, Any]]:
    llm_raw = data.get("llm")
    if not isinstance(llm_raw, dict):
        return []
    models_raw = llm_raw.get("models")
    if not isinstance(models_raw, list):
        return []
    return [item for item in models_raw if isinstance(item, dict)]


def _yaml_model_has_non_empty(
    *, raw_models: list[dict[str, Any]], index: int, field: str
) -> bool:
    if index < 0 or index >= len(raw_models):
        return False
    raw = raw_models[index]
    if field not in raw:
        return False
    return not _is_blank(raw.get(field))


def load_settings(
    path: str | None = None, *, env: dict[str, str] | None = None
) -> AppSettings:
    """Load settings from YAML/JSON and apply env overrides.
    Precedence:
    1) explicit `path`
    2) `SERPSAGE_CONFIG_PATH`
    3) `serpsage.yaml`
    4) defaults
    """
    env_map = env if env is not None else dict(os.environ)
    candidate = path or env_map.get("SERPSAGE_CONFIG_PATH") or "serpsage.yaml"
    p = Path(candidate)
    data: dict[str, Any] = {}
    if p.is_file():
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore[import-untyped] # noqa: PLC0415
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("PyYAML is required to load YAML settings.") from exc
            data = yaml.safe_load(raw) or {}
        elif p.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            raise ValueError(f"Unsupported settings file type: {p.suffix}")
    settings = AppSettings.model_validate(data)
    raw_llm_models = _raw_llm_models(data)
    # Env overrides (centralized; components must not read env).
    provider_backend = str(settings.provider.backend).lower()
    active_provider = settings.provider.resolve_active()
    base_url = (
        env_map.get("PROVIDER_BASE_URL")
        or env_map.get("SEARCH_BASE_URL")
        or (
            env_map.get("GOOGLE_BASE_URL")
            if provider_backend == "google"
            else env_map.get("SEARXNG_BASE_URL")
        )
    )
    api_key = env_map.get("PROVIDER_API_KEY") or env_map.get("SEARCH_API_KEY")
    if base_url:
        active_provider.base_url = base_url
    if api_key:
        active_provider.api_key = api_key
    provider_user_agent = env_map.get("PROVIDER_USER_AGENT") or (
        env_map.get("GOOGLE_USER_AGENT") if provider_backend == "google" else None
    )
    if provider_user_agent:
        active_provider.user_agent = provider_user_agent
    provider_country = env_map.get("PROVIDER_COUNTRY") or (
        env_map.get("GOOGLE_COUNTRY") if provider_backend == "google" else None
    )
    if provider_country:
        active_provider.country = provider_country
    provider_safe = env_map.get("PROVIDER_SAFE") or (
        env_map.get("GOOGLE_SAFE") if provider_backend == "google" else None
    )
    if provider_safe:
        safe_value = str(provider_safe).strip().lower()
        if safe_value == "off":
            active_provider.safe = "off"
        elif safe_value == "medium":
            active_provider.safe = "medium"
        elif safe_value == "high":
            active_provider.safe = "high"
    provider_results_per_page = env_map.get("PROVIDER_RESULTS_PER_PAGE")
    if provider_results_per_page:
        with suppress(Exception):
            active_provider.results_per_page = int(provider_results_per_page)
    # Optional Cloudflare Access headers for providers behind Cloudflare Access.
    cf_id = env_map.get("SEARXNG_CF_ACCESS_CLIENT_ID")
    cf_secret = env_map.get("SEARXNG_CF_ACCESS_CLIENT_SECRET")
    if cf_id and cf_secret:
        active_provider.headers.setdefault("CF-Access-Client-Id", cf_id)
        active_provider.headers.setdefault("CF-Access-Client-Secret", cf_secret)
    # LLM per-backend env fallback.
    # YAML explicit non-empty values always win over env.
    for idx, model in enumerate(settings.llm.models):
        env_api_key_name, env_base_url_name = _llm_env_keys(str(model.backend))
        if env_api_key_name and not _yaml_model_has_non_empty(
            raw_models=raw_llm_models, index=idx, field="api_key"
        ):
            env_api_key = env_map.get(env_api_key_name)
            if env_api_key:
                model.api_key = env_api_key
        if env_base_url_name and not _yaml_model_has_non_empty(
            raw_models=raw_llm_models, index=idx, field="base_url"
        ):
            env_base_url = env_map.get(env_base_url_name)
            if env_base_url:
                model.base_url = env_base_url
    return settings


__all__ = ["load_settings"]
