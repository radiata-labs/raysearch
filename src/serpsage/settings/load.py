from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from serpsage.settings.models import AppSettings


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _overview_env_keys(backend: str) -> tuple[str | None, str | None]:
    key = str(backend or "").strip().lower()
    if key == "openai":
        return "OPENAI_API_KEY", "OPENAI_BASE_URL"
    if key == "gemini":
        return "GEMINI_API_KEY", "GEMINI_BASE_URL"
    return None, None


def _raw_overview_models(data: dict[str, Any]) -> list[dict[str, Any]]:
    overview_raw = data.get("overview")
    if not isinstance(overview_raw, dict):
        return []
    models_raw = overview_raw.get("models")
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
    raw_ov_models = _raw_overview_models(data)

    # Env overrides (centralized; components must not read env).
    base_url = env_map.get("SEARXNG_BASE_URL")
    api_key = env_map.get("SEARCH_API_KEY")
    if base_url:
        settings.provider.searxng.base_url = base_url
    if api_key:
        settings.provider.searxng.api_key = api_key

    # Optional Cloudflare Access headers for searxng (if user uses it).
    cf_id = env_map.get("SEARXNG_CF_ACCESS_CLIENT_ID")
    cf_secret = env_map.get("SEARXNG_CF_ACCESS_CLIENT_SECRET")
    if cf_id and cf_secret:
        settings.provider.searxng.headers.setdefault("CF-Access-Client-Id", cf_id)
        settings.provider.searxng.headers.setdefault(
            "CF-Access-Client-Secret", cf_secret
        )

    # Overview per-backend env fallback.
    # YAML explicit non-empty values always win over env.
    for idx, model in enumerate(settings.overview.models):
        env_api_key_name, env_base_url_name = _overview_env_keys(str(model.backend))
        if env_api_key_name and not _yaml_model_has_non_empty(
            raw_models=raw_ov_models, index=idx, field="api_key"
        ):
            env_api_key = env_map.get(env_api_key_name)
            if env_api_key:
                model.api_key = env_api_key
        if env_base_url_name and not _yaml_model_has_non_empty(
            raw_models=raw_ov_models, index=idx, field="base_url"
        ):
            env_base_url = env_map.get(env_base_url_name)
            if env_base_url:
                model.base_url = env_base_url

    return settings


__all__ = ["load_settings"]
