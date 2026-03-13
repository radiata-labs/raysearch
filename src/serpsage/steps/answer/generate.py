from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any
from typing_extensions import override
from urllib.parse import urlsplit, urlunsplit

from serpsage.components.llm.base import LLMClientBase
from serpsage.dependencies import Depends
from serpsage.models.app.response import AnswerCitation, FetchResultItem
from serpsage.models.steps.answer import (
    AnswerStepContext,
    AnswerSubSearchState,
    PageSource,
    PromptSource,
    QuestionPromptContext,
)
from serpsage.models.steps.search import QuerySourceSpec
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

_CITATION_RE = re.compile(r"\[\s*citation\s*:\s*(\d+)\s*\]", re.IGNORECASE)
_CITATION_GROUP_RE = re.compile(
    r"\[\s*citation\s*:\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)
_FIXED_ABSTRACT_MAX_CHARS = 1000


class AnswerGenerateStep(StepBase[AnswerStepContext]):
    llm: LLMClientBase = Depends()

    @override
    async def run_inner(self, ctx: AnswerStepContext) -> AnswerStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        schema = (
            dict(ctx.request.json_schema)
            if isinstance(ctx.request.json_schema, dict)
            else None
        )
        question_contexts, prompt_sources = self._build_question_prompt_contexts(ctx)
        mode = "json" if schema is not None else "text"
        answer_mode = clean_whitespace(
            str(ctx.plan.answer_mode or "summary")
        ).casefold()
        if answer_mode not in {"direct", "summary"}:
            answer_mode = "summary"
        freshness_intent = bool(ctx.plan.freshness_intent)
        query_language = clean_whitespace(
            str(ctx.plan.query_language or "same as query")
        )
        messages = self._build_messages(
            query=ctx.request.query,
            question_contexts=question_contexts,
            answer_mode=answer_mode,
            mode=mode,
            now_utc=now_utc,
            freshness_intent=freshness_intent,
            query_language=query_language,
        )
        ctx.output.citations = []
        try:
            result = await self.llm.create(
                model=str(self.settings.answer.generate.use_model),
                messages=messages,
                response_format=schema,
            )
            if schema is None:
                answer_text = str(result.text or "")
                normalized_answer = _expand_compound_citation_markers(answer_text)
                ctx.output.answers = normalized_answer
                citation_indexes = _extract_citation_indexes(normalized_answer)
            else:
                raw_answer = (
                    result.data
                    if result.data is not None
                    else _try_parse_json(result.text)
                )
                _validate_json_schema(schema=schema, value=raw_answer)
                normalized_answer = _expand_compound_citation_markers(raw_answer)
                ctx.output.answers = normalized_answer
                citation_indexes = _extract_citation_indexes(normalized_answer)
            ctx.output.citations = self._build_citations(
                sources=prompt_sources,
                citation_indexes=citation_indexes,
                include_content=bool(ctx.request.content),
            )
            ctx.output.answers = _replace_citation_markers(
                ctx.output.answers,
                index_to_id={
                    idx + 1: source.url for idx, source in enumerate(prompt_sources)
                },
            )
            if answer_mode == "direct":
                ctx.output.answers = _strip_citation_markers(ctx.output.answers)
            ctx.output.answers = _sanitize_output_value(ctx.output.answers)
        except _AnswerSchemaMismatchError as exc:
            await self.emit_tracking_event(
                event_name="answer.generate.error",
                request_id=ctx.request_id,
                stage="generate",
                status="error",
                error_code="answer_schema_mismatch",
                error_type=type(exc).__name__,
                attrs={
                    "request_id": ctx.request_id,
                    "message": str(exc),
                },
            )
            ctx.output.answers = {}
            ctx.output.citations = []
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="answer.generate.error",
                request_id=ctx.request_id,
                stage="generate",
                status="error",
                error_code="answer_generate_failed",
                error_type=type(exc).__name__,
                attrs={
                    "request_id": ctx.request_id,
                    "message": str(exc),
                },
            )
            ctx.output.answers = {} if schema is not None else ""
            ctx.output.citations = []
        return ctx

    def _build_question_prompt_contexts(
        self, ctx: AnswerStepContext
    ) -> tuple[list[QuestionPromptContext], list[PromptSource]]:
        sub_searches = self._resolve_sub_searches(ctx)
        question_contexts: list[QuestionPromptContext] = []
        prompt_sources: list[PromptSource] = []
        for question_index, sub_search in enumerate(sub_searches, start=1):
            raw_sources = self._collect_page_sources(sub_search.results)
            local_sources, _ = self._build_prompt_sources(
                sources=raw_sources,
                max_chars=_FIXED_ABSTRACT_MAX_CHARS,
            )
            ordered_sources: list[PromptSource] = []
            for source_index, source in enumerate(local_sources, start=1):
                prompt_source = PromptSource(
                    key=source.key,
                    url=source.url,
                    title=source.title,
                    content=source.content,
                    abstracts=list(source.abstracts),
                    question_index=question_index,
                    source_index=source_index,
                )
                ordered_sources.append(prompt_source)
                prompt_sources.append(prompt_source)
            question_contexts.append(
                QuestionPromptContext(
                    question=clean_whitespace(sub_search.question),
                    sources=ordered_sources,
                )
            )
        return question_contexts, prompt_sources

    def _resolve_sub_searches(
        self, ctx: AnswerStepContext
    ) -> list[AnswerSubSearchState]:
        out: list[AnswerSubSearchState] = []
        for item in list(ctx.search.sub_searches or []):
            question = clean_whitespace(str(item.question or ""))
            search_query = (
                item.search_query.model_copy(deep=True)
                if item.search_query is not None
                else QuerySourceSpec(query=question)
            )
            if not question or not search_query:
                continue
            out.append(
                AnswerSubSearchState(
                    question=question,
                    search_query=search_query,
                    request=(
                        item.request.model_copy(deep=True)
                        if item.request is not None
                        else None
                    ),
                    search_mode=str(item.search_mode or "auto"),
                    results=list(item.results or []),
                )
            )
        if out:
            return out
        fallback_question = clean_whitespace(str(ctx.request.query))
        if not fallback_question:
            return []
        return [
            AnswerSubSearchState(
                question=fallback_question,
                search_query=QuerySourceSpec(query=fallback_question),
                request=(
                    ctx.search.request.model_copy(deep=True)
                    if ctx.search.request is not None
                    else None
                ),
                search_mode=str(ctx.search.search_mode or "auto"),
                results=list(ctx.search.results or []),
            )
        ]

    def _collect_page_sources(self, results: list[FetchResultItem]) -> list[PageSource]:
        sources_by_key: dict[str, PageSource] = {}
        seen_abstracts: dict[str, set[str]] = {}
        page_order = 0
        for result in results:
            page_order = self._append_page_source(
                sources_by_key=sources_by_key,
                seen_abstracts=seen_abstracts,
                page_order=page_order,
                url=str(result.url or ""),
                title=str(result.title or ""),
                content=str(result.content or ""),
                abstracts=list(result.abstracts or []),
            )
            for subpage in list(result.subpages or []):
                page_order = self._append_page_source(
                    sources_by_key=sources_by_key,
                    seen_abstracts=seen_abstracts,
                    page_order=page_order,
                    url=str(subpage.url or ""),
                    title=str(subpage.title or ""),
                    content=str(subpage.content or ""),
                    abstracts=list(subpage.abstracts or []),
                )
        sources = list(sources_by_key.values())
        sources.sort(key=lambda item: item.first_order)
        return sources

    def _append_page_source(
        self,
        *,
        sources_by_key: dict[str, PageSource],
        seen_abstracts: dict[str, set[str]],
        page_order: int,
        url: str,
        title: str,
        content: str,
        abstracts: list[str],
    ) -> int:
        clean_url = clean_whitespace(url)
        key = _normalize_url_key(clean_url)
        if not key:
            return page_order
        normalized_title = clean_whitespace(title)
        source = sources_by_key.get(key)
        if source is None:
            source = PageSource(
                key=key,
                url=clean_url,
                title=normalized_title,
                content=str(content or ""),
                first_order=page_order,
            )
            sources_by_key[key] = source
            seen_abstracts[key] = set()
            page_order += 1
        else:
            if not source.title and normalized_title:
                source.title = normalized_title
            if not source.content and content:
                source.content = str(content)
        for abstract in abstracts:
            text = clean_whitespace(str(abstract or ""))
            if not text:
                continue
            dedupe_key = text.casefold()
            if dedupe_key in seen_abstracts[key]:
                continue
            seen_abstracts[key].add(dedupe_key)
            source.abstracts.append(text)
        return page_order

    def _build_prompt_sources(
        self,
        *,
        sources: list[PageSource],
        max_chars: int,
    ) -> tuple[list[PromptSource], int]:
        budget = max(0, int(max_chars))
        used_chars = 0
        selected_texts: set[str] = set()
        prompt_sources: list[PromptSource] = []
        for source in sources:
            selected_abstracts: list[str] = []
            for abstract in source.abstracts:
                text_key = clean_whitespace(abstract).casefold()
                if not text_key or text_key in selected_texts:
                    continue
                kept_text = abstract
                if budget > 0 and used_chars + len(kept_text) > budget:
                    if used_chars == 0 and not selected_abstracts:
                        kept_text = clean_whitespace(kept_text[:budget])
                        if not kept_text:
                            break
                    else:
                        break
                selected_texts.add(text_key)
                selected_abstracts.append(kept_text)
                used_chars += len(kept_text)
                if budget > 0 and used_chars >= budget:
                    break
            if selected_abstracts:
                prompt_sources.append(
                    PromptSource(
                        key=source.key,
                        url=source.url,
                        title=source.title,
                        content=str(source.content or ""),
                        abstracts=selected_abstracts,
                        question_index=0,
                        source_index=len(prompt_sources) + 1,
                    )
                )
            if budget > 0 and used_chars >= budget:
                break
        return prompt_sources, used_chars

    def _build_messages(
        self,
        *,
        query: str,
        question_contexts: list[QuestionPromptContext],
        answer_mode: str,
        mode: str,
        now_utc: datetime,
        freshness_intent: bool,
        query_language: str,
    ) -> list[dict[str, str]]:
        citation_rules = (
            "Citation format rule: use one citation per tag only, e.g. "
            "[citation:1][citation:2]. Never write comma lists like [citation:1,2]. "
            "Tag syntax must be exact with no extra spaces inside brackets: "
            "use [citation:2], do not write [ citation:2], [citation :2], or [citation: 2 ]."
        )
        quality_rules = (
            "Quality rules:\n"
            f"1) Required output language/script: {query_language}. "
            "Output must strictly match QUERY language/script. This is mandatory.\n"
            "2) Never switch output language based on source language. Translate evidence when needed.\n"
            "3) Use only QUESTION_GROUPS evidence. Do not use outside memory.\n"
            "4) If evidence is weak or missing, explicitly state uncertainty.\n"
            "5) Put citation tags immediately after the supported claim.\n"
            "6) Before finalizing, self-check language compliance.\n"
        )
        question_rules = (
            "Question-group rules:\n"
            "1) QUESTION_GROUPS are retrieval partitions for evidence, not output sections.\n"
            "2) Keep evidence attribution within the same question group; do not cross-mix unsupported claims.\n"
            "3) Use group order internally for reasoning, but do NOT output one section per group.\n"
            "4) Never output labels like Q1/Q2/Question 1 in final answer.\n"
        )
        temporal_rules = (
            "Temporal reasoning rules:\n"
            "1) Determine whether QUERY asks for latest/current/as-of information.\n"
            "2) If yes, choose evidence by explicit date recency (newer date wins).\n"
            "3) For direct latest/current queries, return exactly one best-supported current value with one concrete as-of date.\n"
            "4) Do not mix stale historical values with future projections in one direct answer.\n"
            "5) Do not include forecasts/targets unless QUERY explicitly asks for forecast/target.\n"
            "6) Include a concrete time anchor in the answer, e.g. 'as of YYYY-MM' or exact date.\n"
            "7) If only old/uncertain evidence exists, explicitly say it is the latest known as of that date.\n"
        )
        if mode == "json":
            task = (
                f"{quality_rules}\n"
                f"{question_rules}\n"
                + (f"{temporal_rules}\n" if freshness_intent else "")
                + "Output contract (JSON): return JSON only, no markdown fences, no extra keys. "
                + "For factual claims in string fields, include citations using [citation:x]. "
                + citation_rules
            )
        elif answer_mode == "direct":
            task = (
                f"{quality_rules}\n"
                f"{question_rules}\n"
                + (f"{temporal_rules}\n" if freshness_intent else "")
                + "Output contract (DIRECT): return plain text only. "
                + "Return only the minimal final answer, typically 2-8 words. "
                + "No background or multi-sentence explanation. "
                + "For yes/no questions, start with Yes or No. "
                + "Add a short qualifier only when uncertainty is unavoidable. "
                + "Cite factual claims with [citation:x]. "
                + citation_rules
            )
        else:
            task = (
                f"{quality_rules}\n"
                f"{question_rules}\n"
                + (f"{temporal_rules}\n" if freshness_intent else "")
                + "Output contract (SUMMARY): return markdown only. "
                + "Return one integrated response to QUERY, not one response per sub-question. "
                + "Synthesize evidence into a single coherent conclusion and concise support points. "
                + "Use rich markdown structure when useful: headings, bullet lists, and tables for comparisons/data. "
                + "Do not use code fences unless QUERY explicitly asks for code. "
                + "Do not enumerate by subgroup labels unless QUERY explicitly asks for such structure. "
                + "Every key factual statement should include [citation:x]. "
                + "Do not add a references section. "
                + citation_rules
            )
        question_blocks: list[str] = []
        citation_index = 1
        for q_idx, question_ctx in enumerate(question_contexts, start=1):
            lines = [
                f"[question:{q_idx}]",
                f"question={question_ctx.question}",
                "sources:",
            ]
            if not question_ctx.sources:
                lines.append("- (none)")
            else:
                for source in question_ctx.sources:
                    abstract_lines = (
                        "\n".join(f"- {item}" for item in source.abstracts)
                        if source.abstracts
                        else "- (none)"
                    )
                    lines.extend(
                        [
                            f"[citation:{citation_index}]",
                            f"url={source.url}",
                            f"title={source.title}",
                            "abstracts:",
                            abstract_lines,
                        ]
                    )
                    citation_index += 1
            question_blocks.append("\n".join(lines))
        source_block = "\n\n".join(question_blocks) if question_blocks else "(empty)"
        user_blocks = [
            f"QUERY:\n{query}",
            f"ANSWER_MODE:\n{answer_mode}",
            f"QUESTION_GROUPS:\n{source_block}",
        ]
        if freshness_intent:
            user_blocks = [
                f"CURRENT_UTC_TIMESTAMP:\n{now_utc.isoformat()}",
                f"CURRENT_UTC_DATE:\n{now_utc.date().isoformat()}",
                *user_blocks,
            ]
        user_content = "\n\n".join(user_blocks)
        return [
            {
                "role": "system",
                "content": ("You are a high-precision research assistant. " + task),
            },
            {"role": "user", "content": user_content},
        ]

    def _build_citations(
        self,
        *,
        sources: list[PromptSource],
        citation_indexes: list[int],
        include_content: bool,
    ) -> list[AnswerCitation]:
        out: list[AnswerCitation] = []
        seen_keys: set[str] = set()
        for raw_idx in citation_indexes:
            if raw_idx < 1 or raw_idx > len(sources):
                continue
            source = sources[raw_idx - 1]
            if source.key in seen_keys:
                continue
            seen_keys.add(source.key)
            out.append(
                AnswerCitation(
                    id=source.url,
                    url=source.url,
                    title=source.title,
                    content=(source.content if include_content else None),
                )
            )
        return out


class _AnswerSchemaMismatchError(Exception):
    pass


def _normalize_url_key(url: str) -> str:
    normalized = clean_whitespace(url)
    if not normalized:
        return ""
    try:
        parts = urlsplit(normalized)
    except Exception:  # noqa: BLE001
        return normalized.split("#", 1)[0]
    if not parts.scheme and not parts.netloc:
        return normalized.split("#", 1)[0]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path or "",
            parts.query or "",
            "",
        )
    )


def _extract_citation_indexes(value: object) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()

    def walk(node: object) -> None:
        if isinstance(node, str):
            for match in _CITATION_RE.finditer(node):
                idx = int(match.group(1))
                if idx in seen:
                    continue
                seen.add(idx)
                out.append(idx)
            return
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(value)
    return out


def _expand_compound_citation_markers(value: object) -> object:
    if isinstance(value, str):
        return _expand_compound_citation_markers_in_text(value)
    if isinstance(value, list):
        return [_expand_compound_citation_markers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_compound_citation_markers(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _expand_compound_citation_markers(item) for key, item in value.items()
        }
    return value


def _expand_compound_citation_markers_in_text(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        raw = clean_whitespace(match.group(1) or "")
        indexes = _extract_index_tokens(raw)
        if not indexes:
            return match.group(0)
        return "".join(f"[citation:{idx}]" for idx in indexes)

    return _CITATION_GROUP_RE.sub(_repl, text)


def _extract_index_tokens(raw: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for token in re.findall(r"\d+", raw):
        idx = int(token)
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _replace_citation_markers(value: object, *, index_to_id: dict[int, str]) -> object:
    if not index_to_id:
        return value
    if isinstance(value, str):
        return _replace_citation_markers_in_text(value, index_to_id=index_to_id)
    if isinstance(value, list):
        return [
            _replace_citation_markers(item, index_to_id=index_to_id) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _replace_citation_markers(item, index_to_id=index_to_id) for item in value
        )
    if isinstance(value, dict):
        return {
            key: _replace_citation_markers(item, index_to_id=index_to_id)
            for key, item in value.items()
        }
    return value


def _replace_citation_markers_in_text(text: str, *, index_to_id: dict[int, str]) -> str:
    def _repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        citation_id = index_to_id.get(idx)
        if citation_id is None:
            return match.group(0)
        return f"[citation:{citation_id}]"

    return _CITATION_RE.sub(_repl, text)


def _strip_citation_markers(value: object) -> object:
    if isinstance(value, str):
        return _strip_citation_markers_in_text(value)
    if isinstance(value, list):
        return [_strip_citation_markers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_citation_markers(item) for item in value)
    if isinstance(value, dict):
        return {key: _strip_citation_markers(item) for key, item in value.items()}
    return value


def _strip_citation_markers_in_text(text: str) -> str:
    without = _CITATION_GROUP_RE.sub("", text)
    return without.strip()


def _sanitize_output_value(value: object) -> object:
    if isinstance(value, str):
        return _sanitize_output_text(value)
    if isinstance(value, list):
        return [_sanitize_output_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_output_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _sanitize_output_value(item) for key, item in value.items()}
    return value


def _sanitize_output_text(text: str) -> str:
    cleaned = text.replace("\ufffd\ufffd", "'").replace("\ufffd", "")
    return cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()


def _try_parse_json(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start : end + 1])
        raise


def _validate_json_schema(*, schema: dict[str, Any], value: object) -> None:
    try:
        from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("jsonschema dependency is required") from exc
    try:
        Draft202012Validator(schema).validate(value)
    except Exception as exc:  # noqa: BLE001
        raise _AnswerSchemaMismatchError(str(exc)) from exc


__all__ = ["AnswerGenerateStep"]
