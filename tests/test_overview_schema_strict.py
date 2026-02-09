from __future__ import annotations

import sys


def test_overview_schema_has_additional_properties_false_everywhere():
    # Import via sys.path so this test works without an editable install.
    sys.path.insert(0, "src")
    from serpsage.overview.schema import overview_json_schema  # noqa: PLC0415

    schema = overview_json_schema()

    objs = []

    def walk(node):  # noqa: ANN001
        if isinstance(node, list):
            for x in node:
                walk(x)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            objs.append(node)
        for v in node.values():
            if isinstance(v, (dict, list)):
                walk(v)

    walk(schema)
    assert objs, "expected at least one object schema"
    assert all(o.get("additionalProperties") is False for o in objs)

    # Ensure `$defs` are present for referenced models, otherwise schema is invalid.
    defs = schema.get("$defs") or {}
    assert "Citation" in defs
    assert "OverviewLLMOutput" in defs


def test_overview_schema_strict_required_includes_all_properties():
    sys.path.insert(0, "src")
    from serpsage.overview.schema import overview_json_schema  # noqa: PLC0415

    schema = overview_json_schema()

    def check_obj(obj):  # noqa: ANN001
        if obj.get("type") == "object" or "properties" in obj:
            props = obj.get("properties") or {}
            if props:
                req = obj.get("required") or []
                assert sorted(props.keys()) == sorted(req)

    def walk(node):  # noqa: ANN001
        if isinstance(node, list):
            for x in node:
                walk(x)
            return
        if not isinstance(node, dict):
            return
        check_obj(node)
        for v in node.values():
            if isinstance(v, (dict, list)):
                walk(v)

    walk(schema)
