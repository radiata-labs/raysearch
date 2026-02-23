from __future__ import annotations

import json
from typing import Any

from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.utils import clean_whitespace


def resolve_research_model(
    *, ctx: ResearchStepContext, stage: str, fallback: str
) -> str:
    settings = ctx.settings
    if stage == "plan":
        token = clean_whitespace(settings.research.plan.use_model or "")
    elif stage == "abstract":
        token = clean_whitespace(settings.research.abstract_analyze.use_model or "")
    elif stage == "content":
        token = clean_whitespace(settings.research.content_analyze.use_model or "")
    elif stage == "synthesize":
        token = clean_whitespace(settings.research.synthesize.use_model or "")
    elif stage == "markdown":
        token = clean_whitespace(settings.research.markdown.use_model or "")
    else:
        token = ""
    return token or fallback


async def chat_json(
    *,
    llm,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    retries: int,
) -> dict[str, Any]:
    attempts = max(1, int(retries) + 1)
    payload = list(messages)
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            result = await llm.chat(model=model, messages=payload, schema=schema)
            if result.data is not None:
                raw = result.data
            else:
                text = str(result.text or "")
                if not text:
                    raw = {}
                else:
                    try:
                        raw = json.loads(text)
                    except json.JSONDecodeError:
                        start = text.find("{")
                        end = text.rfind("}")
                        if 0 <= start < end:
                            raw = json.loads(text[start : end + 1])
                        else:
                            raise
            if isinstance(raw, dict):
                return raw
            raise TypeError("LLM JSON output must be an object")
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


def build_abstract_packet(
    *, sources: list[ResearchSource], max_abstracts_per_source: int = 5
) -> str:
    blocks: list[str] = []
    for source in sorted(sources, key=lambda item: item.source_id):
        abstracts = [
            clean_whitespace(x)
            for x in source.abstracts[:max_abstracts_per_source]
            if clean_whitespace(x)
        ]
        abstract_lines = (
            "\n".join(f"- {item}" for item in abstracts) if abstracts else "- (none)"
        )
        blocks.append(
            "\n".join(
                [
                    f"[citation:{source.source_id}]",
                    f"url={source.url}",
                    f"title={clean_whitespace(source.title)}",
                    f"is_subpage={str(bool(source.is_subpage)).lower()}",
                    "abstracts:",
                    abstract_lines,
                ]
            )
        )
    return "\n\n".join(blocks)
