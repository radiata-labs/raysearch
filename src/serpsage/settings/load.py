from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from serpsage.settings.models import AppSettings


def load_settings(path: str | None = None, *, env: dict[str, str] | None = None) -> AppSettings:
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
                import yaml  # type: ignore[import-not-found]
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("PyYAML is required to load YAML settings.") from exc
            data = yaml.safe_load(raw) or {}
        elif p.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            raise ValueError(f"Unsupported settings file type: {p.suffix}")

    settings = AppSettings.model_validate(data)

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

    return settings


__all__ = ["load_settings"]

