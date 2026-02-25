from __future__ import annotations

import traceback
from typing import Any

from serpsage.utils import clean_whitespace

_MAX_CHAIN_DEPTH = 8
_MAX_GROUP_ITEMS = 16
_MAX_TRACEBACK_FRAMES = 5


def exception_summary(exc: BaseException) -> str:
    message = clean_whitespace(str(exc))
    if not message:
        return type(exc).__name__
    return f"{type(exc).__name__}: {message}"


def exception_to_details(
    exc: BaseException,
    *,
    max_chain_depth: int = _MAX_CHAIN_DEPTH,
    max_group_items: int = _MAX_GROUP_ITEMS,
    max_traceback_frames: int = _MAX_TRACEBACK_FRAMES,
) -> dict[str, Any]:
    root = _serialize_exception(
        exc=exc,
        max_traceback_frames=max_traceback_frames,
    )
    chain = _exception_chain(
        exc=exc,
        max_depth=max_chain_depth,
        max_traceback_frames=max_traceback_frames,
    )
    if chain:
        root["chain"] = chain
    group_items = _flatten_exception_group(
        exc=exc,
        max_items=max_group_items,
        max_depth=max_chain_depth,
        max_traceback_frames=max_traceback_frames,
    )
    if group_items:
        root["group_exceptions"] = group_items
    return root


def _exception_chain(
    *,
    exc: BaseException,
    max_depth: int,
    max_traceback_frames: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    node: BaseException | None = exc.__cause__ or exc.__context__
    depth = 0
    while node is not None and depth < max(1, int(max_depth)):
        node_id = id(node)
        if node_id in seen:
            break
        seen.add(node_id)
        out.append(
            _serialize_exception(
                exc=node,
                max_traceback_frames=max_traceback_frames,
            )
        )
        node = node.__cause__ or node.__context__
        depth += 1
    return out


def _flatten_exception_group(
    *,
    exc: BaseException,
    max_items: int,
    max_depth: int,
    max_traceback_frames: int,
) -> list[dict[str, Any]]:
    limit = max(1, int(max_items))
    out: list[dict[str, Any]] = []
    stack: list[tuple[BaseException, tuple[int, ...], int]] = [(exc, (), 0)]
    while stack and len(out) < limit:
        node, path, depth = stack.pop()
        if depth > max(1, int(max_depth)):
            continue
        children = _group_children(node)
        if children is None:
            if not path:
                continue
            item = _serialize_exception(
                exc=node,
                max_traceback_frames=max_traceback_frames,
            )
            item["path"] = ".".join(str(idx) for idx in path)
            out.append(item)
            continue
        for child_index in range(len(children) - 1, -1, -1):
            child = children[child_index]
            if isinstance(child, BaseException):
                stack.append((child, path + (child_index,), depth + 1))
    return out


def _serialize_exception(
    *,
    exc: BaseException,
    max_traceback_frames: int,
) -> dict[str, Any]:
    message = clean_whitespace(str(exc))
    out: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": message,
    }
    attrs = _safe_exception_attrs(exc)
    if attrs:
        out["attrs"] = attrs
    frames = _traceback_tail(exc=exc, limit=max_traceback_frames)
    if frames:
        out["traceback_tail"] = frames
    direct_cause = exc.__cause__ or exc.__context__
    if direct_cause is not None:
        out["cause"] = {
            "type": type(direct_cause).__name__,
            "message": clean_whitespace(str(direct_cause)),
        }
    return out


def _traceback_tail(*, exc: BaseException, limit: int) -> list[dict[str, Any]]:
    tb = exc.__traceback__
    if tb is None:
        return []
    count = max(1, int(limit))
    frames = traceback.extract_tb(tb)[-count:]

    return [
        {
            "file": str(frame.filename),
            "line": int(frame.lineno) if frame.lineno is not None else 0,
            "func": str(frame.name),
        }
        for frame in frames
    ]


def _safe_exception_attrs(exc: BaseException) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    raw = getattr(exc, "__dict__", None)
    if not isinstance(raw, dict):
        return attrs
    for key, value in raw.items():
        name = str(key)
        if not name or name.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            attrs[name] = value
            continue
        if isinstance(value, (list, tuple)):
            if len(value) > 8:
                continue
            if all(
                isinstance(item, (str, int, float, bool)) or item is None
                for item in value
            ):
                attrs[name] = list(value)
    return attrs


def _group_children(exc: BaseException) -> tuple[Any, ...] | None:
    children = getattr(exc, "exceptions", None)
    if isinstance(children, tuple):
        return children
    return None


__all__ = ["exception_summary", "exception_to_details"]
