from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from pydantic import BaseModel

from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase

ChatMessage: TypeAlias = dict[str, str]
TModel = TypeVar("TModel", bound=BaseModel)


def resolve_research_model(
    *, ctx: ResearchStepContext, stage: str, fallback: str
) -> str:
    model_settings = ctx.settings.research.models
    stage_to_model = {
        "plan": model_settings.plan,
        "abstract": model_settings.abstract_analyze,
        "content": model_settings.content_analyze,
        "synthesize": model_settings.synthesize,
        "markdown": model_settings.markdown,
    }
    token = clean_whitespace(stage_to_model.get(stage, ""))
    return token or fallback


async def chat_pydantic(
    *,
    llm: LLMClientBase,
    model: str,
    messages: list[ChatMessage],
    schema_model: type[TModel],
    retries: int,
    schema_json: dict[str, Any] | None = None,
) -> TModel:
    attempts = max(1, int(retries) + 1)
    payload = list(messages)
    last_exc: Exception | None = None
    schema = (
        dict(schema_json)
        if isinstance(schema_json, dict)
        else schema_model.model_json_schema()
    )
    for _ in range(attempts):
        try:
            result = await llm.chat(model=model, messages=payload, schema=schema)
            raw = _decode_json_payload(result.data, result.text)
            return schema_model.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
            payload = payload + [
                {
                    "role": "user",
                    "content": (
                        "The previous output was invalid. Return JSON only and strictly match "
                        "the schema. Do not include comments, markdown fences, or extra keys."
                    ),
                }
            ]
    assert last_exc is not None
    raise last_exc


def normalize_strings(raw: object, *, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = clean_whitespace(str(item or ""))
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max(1, int(limit)):
            break
    return out


def merge_strings(*groups: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            value = clean_whitespace(str(item or ""))
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            if len(out) >= max(1, int(limit)):
                return out
    return out


def _decode_json_payload(data: object | None, text: str) -> object:
    if data is not None:
        return data
    raw_text = str(text or "")
    if not raw_text:
        return {}
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if 0 <= start < end:
            return json.loads(raw_text[start : end + 1])
        raise


def build_abstract_packet(
    *, sources: list[ResearchSource], max_abstracts_per_source: int = 5
) -> str:
    blocks: list[str] = []
    for source in sorted(sources, key=lambda item: item.source_id):
        abstracts = [
            text
            for x in source.abstracts[:max_abstracts_per_source]
            if (text := _normalize_block_text(str(x)))
        ]
        abstract_lines: list[str] = []
        if abstracts:
            for index, item in enumerate(abstracts, start=1):
                if "\n" in item:
                    abstract_lines.extend(
                        [
                            f"  - Abstract {index}:",
                            "    ```text",
                            *[f"    {line}" for line in item.split("\n")],
                            "    ```",
                        ]
                    )
                else:
                    abstract_lines.append(f"  - {item}")
        else:
            abstract_lines.append("  - (none)")
        blocks.append(
            "\n".join(
                [
                    f"### Source {int(source.source_id)}",
                    f"- URL: {source.url}",
                    f"- Title: {clean_whitespace(source.title)}",
                    f"- Is subpage: {str(bool(source.is_subpage)).lower()}",
                    "- Abstracts:",
                    "\n".join(abstract_lines),
                ]
            )
        )
    return "\n\n".join(blocks)


def _normalize_block_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
