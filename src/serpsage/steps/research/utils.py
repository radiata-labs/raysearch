from __future__ import annotations

import json
import re
from typing import Any

from serpsage.app.response import FetchResultItem, FetchSubpagesResult
from serpsage.models.errors import AppError
from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.utils import clean_whitespace, strip_html

_CITATION_GROUP_RE = re.compile(
    r"\[\s*citation\s*:\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)
_CITATION_SINGLE_RE = re.compile(
    r"\[\s*citation\s*:\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)


def resolve_research_model(*, ctx: ResearchStepContext, stage: str, fallback: str) -> str:
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
    ctx: ResearchStepContext,
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
            raw = result.data if result.data is not None else try_parse_json_value(result.text)
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


def try_parse_json_value(text: str) -> object:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start : end + 1])
        raise


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


def normalize_source_ids(raw: object, *, limit: int) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in raw:
        try:
            value = int(item)
        except Exception:  # noqa: S112
            continue
        if value <= 0:
            continue
        if value in seen:
            continue
        seen.add(value)
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


def upsert_source_from_fetch_result(
    *,
    ctx: ResearchStepContext,
    result: FetchResultItem,
    round_index: int,
) -> list[int]:
    created: list[int] = []
    source_id, is_new = upsert_source(
        ctx=ctx,
        url=str(result.url),
        title=str(result.title),
        abstracts=list(result.abstracts or []),
        content=str(result.content or ""),
        round_index=round_index,
        parent_url="",
        is_subpage=False,
    )
    if is_new:
        created.append(source_id)
    for sub in list(result.subpages or []):
        sub_id, sub_is_new = upsert_source_from_subpage(
            ctx=ctx,
            parent_url=str(result.url),
            sub=sub,
            round_index=round_index,
        )
        if sub_is_new:
            created.append(sub_id)
    return created


def upsert_source_from_subpage(
    *,
    ctx: ResearchStepContext,
    parent_url: str,
    sub: FetchSubpagesResult,
    round_index: int,
) -> tuple[int, bool]:
    return upsert_source(
        ctx=ctx,
        url=str(sub.url),
        title=str(sub.title),
        abstracts=list(sub.abstracts or []),
        content=str(sub.content or ""),
        round_index=round_index,
        parent_url=parent_url,
        is_subpage=True,
    )


def upsert_source(
    *,
    ctx: ResearchStepContext,
    url: str,
    title: str,
    abstracts: list[str],
    content: str,
    round_index: int,
    parent_url: str,
    is_subpage: bool,
) -> tuple[int, bool]:
    existing = ctx.corpus.source_url_to_id.get(url)
    if existing is not None:
        for source in ctx.corpus.sources:
            if source.source_id != existing:
                continue
            if not source.title and title:
                source.title = title
            source.abstracts = merge_strings(
                list(source.abstracts),
                normalize_strings(abstracts, limit=32),
                limit=32,
            )
            if not source.content and content:
                source.content = content
            return existing, False

    source_id = len(ctx.corpus.sources) + 1
    ctx.corpus.sources.append(
        ResearchSource(
            source_id=source_id,
            url=url,
            title=title,
            abstracts=normalize_strings(abstracts, limit=32),
            content=content,
            round_index=round_index,
            parent_url=parent_url,
            is_subpage=bool(is_subpage),
        )
    )
    ctx.corpus.source_url_to_id[url] = source_id
    return source_id, True


def build_abstract_packet(*, sources: list[ResearchSource], max_abstracts_per_source: int = 5) -> str:
    blocks: list[str] = []
    for source in sorted(sources, key=lambda item: item.source_id):
        abstracts = [clean_whitespace(x) for x in source.abstracts[:max_abstracts_per_source] if clean_whitespace(x)]
        abstract_lines = "\n".join(f"- {item}" for item in abstracts) if abstracts else "- (none)"
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


def build_content_packet(*, sources: list[ResearchSource], source_ids: list[int], max_chars: int = 6000) -> str:
    wanted = set(source_ids)
    blocks: list[str] = []
    for source in sorted(sources, key=lambda item: item.source_id):
        if source.source_id not in wanted:
            continue
        content = clean_whitespace(str(source.content or ""))
        if len(content) > max_chars:
            content = content[:max_chars]
        blocks.append(
            "\n".join(
                [
                    f"[citation:{source.source_id}]",
                    f"url={source.url}",
                    f"title={clean_whitespace(source.title)}",
                    "content:",
                    content or "(empty)",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_round_notes(ctx: ResearchStepContext, *, limit: int = 24) -> str:
    chunks = [clean_whitespace(item) for item in ctx.notes if clean_whitespace(item)]
    if not chunks:
        return "(none)"
    return "\n".join(f"- {item}" for item in chunks[-limit:])


def replace_numeric_citations_with_urls(
    text: str,
    *,
    index_to_url: dict[int, str],
) -> tuple[str, list[int]]:
    invalid: list[int] = []
    invalid_seen: set[int] = set()

    def repl(match: re.Match[str]) -> str:
        raw = clean_whitespace(match.group(1) or "")
        idx_tokens = re.findall(r"\d+", raw)
        if not idx_tokens:
            return ""
        out: list[str] = []
        for token in idx_tokens:
            idx = int(token)
            url = index_to_url.get(idx)
            if not url:
                if idx not in invalid_seen:
                    invalid_seen.add(idx)
                    invalid.append(idx)
                continue
            out.append(f"[citation:{url}]")
        return "".join(out)

    return _CITATION_GROUP_RE.sub(repl, text), invalid


def strip_citation_markers(value: object) -> tuple[object, int]:
    removed = 0

    def walk(node: object) -> object:
        nonlocal removed
        if isinstance(node, str):
            count = len(_CITATION_SINGLE_RE.findall(node))
            if count:
                removed += count
            return _CITATION_SINGLE_RE.sub("", node).strip()
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        if isinstance(node, dict):
            return {key: walk(item) for key, item in node.items()}
        return node

    return walk(value), removed


def normalize_markdown(text: str) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def add_error(ctx: ResearchStepContext, *, code: str, message: str, details: dict[str, Any]) -> None:
    ctx.errors.append(AppError(code=code, message=message, details=details))


def normalize_search_text(value: str) -> str:
    return clean_whitespace(strip_html(str(value or "")))
