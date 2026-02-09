from __future__ import annotations

import json
from typing import Any


def stable_json(obj: Any) -> str:
    """Deterministic JSON representation used for cache keys."""

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["stable_json"]
