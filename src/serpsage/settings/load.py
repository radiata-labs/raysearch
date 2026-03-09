from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from serpsage.settings.models import AppSettings


def load_settings(
    path: str | None = None, *, env: dict[str, str] | None = None
) -> AppSettings:
    """Load settings from YAML/JSON and preserve runtime env for component injection.

    Precedence:
    1) explicit `path`
    2) `SERPSAGE_CONFIG_PATH`
    3) `serpsage.yaml`
    4) defaults

    Raw component instance declarations are preserved through validation so the
    isolated component loader can distinguish explicit user config from
    post-merge defaults.
    """
    env_map = dict(env) if env is not None else dict(os.environ)
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
    settings.runtime_env = env_map
    return settings


__all__ = ["load_settings"]
