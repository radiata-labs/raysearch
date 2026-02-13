from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic.json_schema import models_json_schema

from serpsage.app.response import OverviewLLMOutput


def overview_json_schema() -> dict[str, Any]:
    # NOTE:
    # - Some OpenAI-compatible providers (and OpenAI strict mode) validate JSON
    #   schema more strictly than Pydantic's defaults.
    # - In strict mode, OpenAI requires:
    #   1) `additionalProperties: false` for every object schema
    #   2) `required` must include *all* keys in `properties` for every object schema
    #
    # Also, Pydantic's `Model.model_json_schema()` can yield a `$ref` to `$defs`
    # with an empty `$defs` depending on context; `models_json_schema()` is a
    # more reliable way to obtain a complete, self-contained schema.
    _, defs = models_json_schema([(OverviewLLMOutput, "validation")])
    defs_map: dict[str, Any] = dict(defs.get("$defs") or {})

    root = dict(defs_map.get("OverviewLLMOutput") or {})
    if not root:
        # Fallback (should not happen): keep backward compatibility.
        root = OverviewLLMOutput.model_json_schema()

    base: dict[str, Any] = {"$defs": defs_map, **root}
    out: dict[str, Any] = deepcopy(base)

    def walk(node: object) -> None:
        if isinstance(node, list):
            for x in node:
                walk(x)
            return
        if not isinstance(node, dict):
            return

        # Treat any schema with properties as an object schema (even if `type`
        # is omitted due to `$ref` or composition).
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            props = node.get("properties")
            if isinstance(props, dict) and props:
                # OpenAI strict requires required to include all properties.
                node["required"] = sorted(str(k) for k in props)

        for v in node.values():
            if isinstance(v, (dict, list)):
                walk(v)

    walk(out)
    return out


__all__ = ["overview_json_schema"]
