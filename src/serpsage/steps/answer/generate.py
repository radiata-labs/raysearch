from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import urlsplit, urlunsplit

from serpsage.app.response import AnswerCitation, FetchResultItem
from serpsage.models.errors import AppError
from serpsage.models.pipeline import AnswerStepContext
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


_CITATION_RE = re.compile(r"\[\s*citation\s*:\s*(\d+)\s*\]", re.IGNORECASE)
_CITATION_GROUP_RE = re.compile(
    r"\[\s*citation\s*:\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)


@dataclass(slots=True)
class _PageAbstract:
    score: float
    order: int
    text: str


@dataclass(slots=True)
class _PageSource:
    key: str
    url: str
    title: str
    content: str
    first_order: int
    max_score: float = 0.0
    abstracts: list[_PageAbstract] = field(default_factory=list)


@dataclass(slots=True)
class _GlobalAbstract:
    page_key: str
    score: float
    order: int
    text: str


@dataclass(slots=True)
class _PromptSource:
    key: str
    url: str
    title: str
    content: str
    abstracts: list[str]
    score: float
    first_order: int


class AnswerGenerateStep(StepBase[AnswerStepContext]):
    span_name = "step.answer_generate"

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(
        self, ctx: AnswerStepContext, *, span: SpanBase
    ) -> AnswerStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        schema = (
            dict(ctx.request.json_schema)
            if isinstance(ctx.request.json_schema, dict)
            else None
        )
        if any(err.code == "search_query_expansion_failed" for err in ctx.errors):
            ctx.errors.append(
                AppError(
                    code="answer_generate_skipped",
                    message="search aborted before answer generation",
                    details={
                        "request_id": ctx.request_id,
                        "stage": "generate",
                        "reason": "deep_search_aborted",
                    },
                )
            )
            ctx.output.answers = {} if schema is not None else ""
            ctx.output.citations = []
            span.set_attr("source_pages", 0)
            span.set_attr("source_abstract_chars", 0)
            span.set_attr("citation_count", 0)
            span.set_attr("freshness_intent", bool(ctx.plan.freshness_intent))
            span.set_attr("schema_mode", bool(schema is not None))
            span.set_attr("skipped", True)
            span.set_attr("skip_reason", "deep_search_aborted")
            return ctx

        raw_sources = self._collect_page_sources(ctx.search.results)
        prompt_sources, abstract_chars = self._build_prompt_sources(
            sources=raw_sources,
            max_chars=int(self.settings.answer.generate.max_abstract_chars),
        )

        mode = "json" if schema is not None else "text"
        answer_mode = (
            "direct"
            if clean_whitespace(str(ctx.plan.answer_mode or "")).casefold() == "direct"
            else "summary"
        )
        freshness_intent = bool(ctx.plan.freshness_intent)
        query_language = clean_whitespace(
            str(ctx.plan.query_language or "same as query")
        )
        messages = self._build_messages(
            query=ctx.request.query,
            sources=prompt_sources,
            answer_mode=answer_mode,
            mode=mode,
            now_utc=now_utc,
            freshness_intent=freshness_intent,
            query_language=query_language,
        )
        ctx.output.citations = []

        try:
            result = await self._llm.chat(
                model=str(self.settings.answer.generate.use_model),
                messages=messages,
                schema=schema,
            )
            if schema is None:
                answer_text = clean_whitespace(result.text or "")
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
            ctx.errors.append(
                AppError(
                    code="answer_schema_mismatch",
                    message=str(exc),
                    details={"request_id": ctx.request_id, "stage": "generate"},
                )
            )
            ctx.output.answers = {}
            ctx.output.citations = []
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="answer_generate_failed",
                    message=str(exc),
                    details={"request_id": ctx.request_id, "stage": "generate"},
                )
            )
            ctx.output.answers = {} if schema is not None else ""
            ctx.output.citations = []

        span.set_attr("source_pages", int(len(prompt_sources)))
        span.set_attr("source_abstract_chars", int(abstract_chars))
        span.set_attr("citation_count", int(len(ctx.output.citations)))
        span.set_attr("freshness_intent", bool(freshness_intent))
        span.set_attr("schema_mode", bool(schema is not None))
        span.set_attr("skipped", False)
        return ctx

    def _collect_page_sources(
        self, results: list[FetchResultItem]
    ) -> list[_PageSource]:
        sources_by_key: dict[str, _PageSource] = {}
        seen_abstracts: dict[str, set[str]] = {}
        page_order = 0
        abstract_order = 0

        for result in results:
            page_order, abstract_order = self._append_page_source(
                sources_by_key=sources_by_key,
                seen_abstracts=seen_abstracts,
                page_order=page_order,
                abstract_order=abstract_order,
                url=str(result.url or ""),
                title=str(result.title or ""),
                content=str(result.content or ""),
                abstracts=list(result.abstracts or []),
                scores=list(result.abstract_scores or []),
            )
            for subpage in list(result.subpages or []):
                page_order, abstract_order = self._append_page_source(
                    sources_by_key=sources_by_key,
                    seen_abstracts=seen_abstracts,
                    page_order=page_order,
                    abstract_order=abstract_order,
                    url=str(subpage.url or ""),
                    title=str(subpage.title or ""),
                    content=str(subpage.content or ""),
                    abstracts=list(subpage.abstracts or []),
                    scores=list(subpage.abstract_scores or []),
                )

        sources = list(sources_by_key.values())
        sources.sort(key=lambda item: (-item.max_score, item.first_order))
        return sources

    def _append_page_source(
        self,
        *,
        sources_by_key: dict[str, _PageSource],
        seen_abstracts: dict[str, set[str]],
        page_order: int,
        abstract_order: int,
        url: str,
        title: str,
        content: str,
        abstracts: list[str],
        scores: list[float],
    ) -> tuple[int, int]:
        clean_url = clean_whitespace(url)
        key = _normalize_url_key(clean_url)
        if not key:
            return page_order, abstract_order

        source = sources_by_key.get(key)
        if source is None:
            source = _PageSource(
                key=key,
                url=clean_url,
                title=clean_whitespace(title),
                content=str(content or ""),
                first_order=page_order,
            )
            sources_by_key[key] = source
            seen_abstracts[key] = set()
            page_order += 1
        else:
            if not source.title and clean_whitespace(title):
                source.title = clean_whitespace(title)
            if not source.content and content:
                source.content = str(content)

        count = max(len(abstracts), len(scores))
        for idx in range(count):
            text = clean_whitespace(str(abstracts[idx] if idx < len(abstracts) else ""))
            if not text:
                continue
            dedupe_key = text.casefold()
            if dedupe_key in seen_abstracts[key]:
                continue
            seen_abstracts[key].add(dedupe_key)
            score = float(scores[idx]) if idx < len(scores) else 0.0
            source.abstracts.append(
                _PageAbstract(score=score, order=abstract_order, text=text)
            )
            source.max_score = max(source.max_score, score)
            abstract_order += 1

        return page_order, abstract_order

    def _build_prompt_sources(
        self,
        *,
        sources: list[_PageSource],
        max_chars: int,
    ) -> tuple[list[_PromptSource], int]:
        global_abstracts: list[_GlobalAbstract] = []
        for source in sources:
            global_abstracts.extend(
                _GlobalAbstract(
                    page_key=source.key,
                    score=float(item.score),
                    order=int(item.order),
                    text=item.text,
                )
                for item in source.abstracts
            )
        global_abstracts.sort(key=lambda item: (-item.score, item.order))

        budget = max(0, int(max_chars))
        used_chars = 0
        selected_texts: set[str] = set()
        selected_by_page: dict[str, list[_GlobalAbstract]] = {}

        for item in global_abstracts:
            text_key = clean_whitespace(item.text).casefold()
            if not text_key or text_key in selected_texts:
                continue
            kept_text = item.text
            if budget > 0 and used_chars + len(kept_text) > budget:
                if used_chars == 0:
                    kept_text = clean_whitespace(kept_text[:budget])
                    if not kept_text:
                        break
                else:
                    continue
            selected_texts.add(text_key)
            selected_by_page.setdefault(item.page_key, []).append(
                _GlobalAbstract(
                    page_key=item.page_key,
                    score=item.score,
                    order=item.order,
                    text=kept_text,
                )
            )
            used_chars += len(kept_text)
            if budget > 0 and used_chars >= budget:
                break

        prompt_sources: list[_PromptSource] = []
        for source in sources:
            selected = selected_by_page.get(source.key) or []
            if not selected:
                continue
            selected.sort(key=lambda item: (-item.score, item.order))
            prompt_sources.append(
                _PromptSource(
                    key=source.key,
                    url=source.url,
                    title=source.title,
                    content=str(source.content or ""),
                    abstracts=[item.text for item in selected],
                    score=max((item.score for item in selected), default=0.0),
                    first_order=source.first_order,
                )
            )

        prompt_sources.sort(key=lambda item: (-item.score, item.first_order))
        return prompt_sources, used_chars

    def _build_messages(
        self,
        *,
        query: str,
        sources: list[_PromptSource],
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
            "2) Never switch output language based on SOURCE_PAGES language. "
            "When sources use other languages, translate the evidence into required output language.\n"
            "3) Never output mixed-language sentences unless the QUERY itself is mixed-language.\n"
            "4) Use only SOURCE_PAGES evidence. Do not use outside memory.\n"
            "5) Prefer higher-score sources when choosing between claims.\n"
            "6) If multiple sources conflict, state the conflict briefly and prefer better-supported evidence.\n"
            "7) If evidence is weak or missing, explicitly state uncertainty.\n"
            "8) Put citation tags immediately after the supported claim.\n"
            "9) Before finalizing, self-check language compliance; if any sentence is not in required language/script, rewrite it.\n"
            "10) Do not copy mojibake/corrupted characters (such as �, ��). If a source segment is corrupted, omit it or restate cleanly.\n"
        )
        temporal_rules = (
            "Temporal reasoning rules:\n"
            "1) Determine whether QUERY asks for latest/current/as-of information.\n"
            "2) If yes, choose evidence by explicit date recency (newer date wins).\n"
            "3) For direct latest/current queries, return exactly one best-supported current value with one concrete as-of date.\n"
            "4) Do not mix stale historical values with future projections in one direct answer.\n"
            "5) Do not include forecasts/targets/IPO goals unless QUERY explicitly asks for forecast or target.\n"
            "6) Include a concrete time anchor in the answer, e.g. 'as of YYYY-MM' or exact date.\n"
            "7) If only old/uncertain evidence exists, explicitly say it is the latest known as of that date.\n"
        )
        if mode == "json":
            task = (
                f"{quality_rules}\n"
                + (f"{temporal_rules}\n" if freshness_intent else "")
                + "Output contract (JSON): return JSON only, no markdown fences, no extra keys. "
                + "For factual claims in string fields, include citations using [citation:x]. "
                + citation_rules
            )
        elif answer_mode == "direct":
            task = (
                f"{quality_rules}\n"
                + (f"{temporal_rules}\n" if freshness_intent else "")
                + "Output contract (DIRECT): return plain text only. "
                + "Start with the direct answer in the first sentence. "
                + "For concrete factual queries, answer directly (e.g. Paris, $1.5 trillion). "
                + "Use exactly 1 sentence when evidence is sufficient; at most 2 only when uncertainty must be stated. "
                + "Do not add background, comparisons, or side facts unless QUERY asks for them. "
                + "If QUERY asks for latest/current value (valuation/price/leader/etc.), output only the single current value with its as-of date; do not append future plans, targets, or projections. "
                + "Cite factual claims with [citation:x]. "
                + citation_rules
            )
        else:
            task = (
                f"{quality_rules}\n"
                + (f"{temporal_rules}\n" if freshness_intent else "")
                + "Output contract (SUMMARY): return plain text only. "
                + "First give a one-sentence conclusion, then 3-6 concise key points. "
                + "Every key factual statement should include [citation:x]. "
                + "Do not add a references section. "
                + citation_rules
            )

        source_blocks: list[str] = []
        for idx, source in enumerate(sources, 1):
            abstract_lines = (
                "\n".join(f"- {item}" for item in source.abstracts)
                if source.abstracts
                else "- (none)"
            )
            source_blocks.append(
                "\n".join(
                    [
                        f"[citation:{idx}]",
                        f"score={source.score:.4f}",
                        f"url={source.url}",
                        f"title={source.title}",
                        "abstracts:",
                        abstract_lines,
                    ]
                )
            )
        source_block = "\n\n".join(source_blocks) if source_blocks else "(empty)"
        user_blocks = [
            f"QUERY:\n{query}",
            f"ANSWER_MODE:\n{answer_mode}",
            f"SOURCE_PAGES:\n{source_block}",
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
        sources: list[_PromptSource],
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
        if isinstance(node, list | tuple):
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
    return clean_whitespace(without)


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
    return clean_whitespace(cleaned)


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
        from jsonschema import Draft202012Validator
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("jsonschema dependency is required") from exc
    try:
        Draft202012Validator(schema).validate(value)
    except Exception as exc:  # noqa: BLE001
        raise _AnswerSchemaMismatchError(str(exc)) from exc


__all__ = ["AnswerGenerateStep"]
