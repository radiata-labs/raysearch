from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_settings(
    path: str | None = None, *, env: dict[str, str] | None = None
) -> dict[str, Any]:
    """Load raw settings data from YAML/JSON and preserve process env values.

    Precedence:
    1) explicit `path`
    2) `RAYSEARCH_CONFIG_PATH`
    3) `raysearch.yaml`
    4) defaults

    The final AppSettings model is built later, after component discovery has
    injected component-specific config models into the settings schema.
    """
    env_map = dict(env) if env is not None else dict(os.environ)
    candidate = path or env_map.get("RAYSEARCH_CONFIG_PATH") or "raysearch.yaml"
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
    data["runtime_env"] = env_map
    return data


__all__ = ["load_settings"]
