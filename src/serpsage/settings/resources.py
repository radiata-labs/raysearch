from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class ResourceError(RuntimeError):
    pass


def load_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise ResourceError(f"Resource not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if "\ufffd" in raw:
        raise ResourceError(f"Resource contains replacement characters: {path}")
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


__all__ = ["ResourceError", "load_lines"]
