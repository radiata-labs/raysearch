from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.dependencies import Depends
from serpsage.models.steps.search import (
    SearchQueryCandidate,
    SearchQueryJob,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from collections.abc import Sequence

_RE_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_KANA = re.compile(r"[\u3040-\u30ff]")
_RE_VERSION_LIKE_TOKEN = re.compile(r"(?i)^[a-z]*\d+(?:[._-]\d+)+(?:[a-z0-9._-]*)$")
_ZH_INTENT_SUFFIX = "\u5b98\u65b9 \u6587\u6863 \u6307\u5357 \u5bf9\u6bd4"
_ZH_EVIDENCE_SUFFIX = "\u8bc4\u6d4b \u62a5\u544a \u6765\u6e90"
_JA_INTENT_SUFFIX = (
    "\u516c\u5f0f \u30c9\u30ad\u30e5\u30e1\u30f3\u30c8 \u30ac\u30a4\u30c9 \u6bd4\u8f03"
)
_JA_EVIDENCE_SUFFIX = (
    "\u30d9\u30f3\u30c1\u30de\u30fc\u30af \u30ec\u30dd\u30fc\u30c8 \u30bd\u30fc\u30b9"
)
_EN_INTENT_SUFFIX = "official docs guide comparison"
_EN_EVIDENCE_SUFFIX = "benchmark report source"


class SearchExpandStep(StepBase[SearchStepContext]):
    llm: LLMClientBase = Depends()

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        ctx.plan.aborted = False
        ctx.plan.abort_reason = ""
        ctx.plan.query_jobs = []
        primary_query = clean_whitespace(str(ctx.request.query or ""))
        ctx.plan.optimized_query = clean_whitespace(
            str(ctx.plan.optimized_query or primary_query)
        )
        if not primary_query:
            await self._abort_empty_query(ctx=ctx, raw_query=ctx.request.query)
            return ctx
        max_extra_queries = max(0, int(ctx.plan.max_extra_queries))
        if max_extra_queries <= 0:
            ctx.plan.query_jobs = [
                SearchQueryJob(query=primary_query, source="primary")
            ]
            return ctx
        manual_queries = (
            self._normalize_queries(list(ctx.request.additional_queries or []))
            if ctx.plan.mode == "deep"
            else []
        )
        rule_queries = self._build_rule_queries(query=primary_query)
        if (
            ctx.plan.mode == "deep"
            and max_extra_queries > 0
            and not bool(ctx.runtime.disable_internal_llm)
        ):
            llm_queries = await self._collect_llm_queries(
                ctx=ctx,
                query=primary_query,
                max_queries=max_extra_queries,
            )
        else:
            llm_queries = []
        ctx.plan.query_jobs = self._merge_query_jobs(
            primary_query=primary_query,
            manual_queries=manual_queries,
            rule_queries=rule_queries,
            llm_queries=llm_queries,
            max_extra_queries=max_extra_queries,
        )
        return ctx

    async def _abort_empty_query(
        self, *, ctx: SearchStepContext, raw_query: object
    ) -> None:
        ctx.plan.aborted = True
        ctx.plan.abort_reason = "empty query"
        await self.emit_tracking_event(
            event_name="search.expand.error",
            request_id=ctx.request_id,
            stage="search_expand",
            status="error",
            error_code="search_query_expansion_failed",
            attrs={
                "query": str(raw_query),
                "message": "empty query",
            },
        )

    async def _collect_llm_queries(
        self,
        *,
        ctx: SearchStepContext,
        query: str,
        max_queries: int,
    ) -> list[str]:
        if max_queries <= 0:
            return []
        model_name = self._resolve_expansion_model()
        timeout_s = float(self.settings.search.expansion.llm_timeout_s)
        try:
            return await self._expand_with_llm(
                query=query,
                model_name=model_name,
                max_queries=max_queries,
                timeout_s=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="search.expand.error",
                request_id=ctx.request_id,
                stage="search_expand",
                status="error",
                error_code="search_query_expansion_failed",
                error_type=type(exc).__name__,
                attrs={
                    "query": query,
                    "model": model_name,
                    "message": str(exc),
                    "fallback": "primary_manual_rule",
                },
            )
            return []

    def _resolve_expansion_model(self) -> str:
        model_name = clean_whitespace(
            str(self.settings.search.expansion.llm_model or "")
        )
        if model_name:
            return model_name
        return str(self.settings.answer.plan.use_model)

    async def _expand_with_llm(
        self,
        *,
        query: str,
        model_name: str,
        max_queries: int,
        timeout_s: float,
    ) -> list[str]:
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["queries"],
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": int(max_queries),
                }
            },
        }
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a query expansion engine for deep web retrieval. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        f"Original query: {query}",
                        f"Generate up to {max_queries} additional search queries.",
                        "Requirements:",
                        "- Keep same language/script as original query.",
                        "- Do not translate query language into another language.",
                        "- For Chinese/Japanese queries, keep outputs in Chinese/Japanese except named entities.",
                        "- Keep each query concise and web-search friendly.",
                        "- Make each query semantically distinct from others.",
                        "- Include time/entity qualifiers only when implied by original query.",
                        'Output JSON: {"queries": ["...", "..."]}',
                    ]
                ),
            },
        ]
        result = await self.llm.create(
            model=model_name,
            messages=messages,
            response_format=schema,
            timeout_s=timeout_s,
        )
        raw = (
            result.data
            if result.data is not None
            else self._try_parse_json_value(result.text)
        )
        if not isinstance(raw, dict):
            raise TypeError("query expansion output must be an object")
        raw_queries = raw.get("queries")
        if not isinstance(raw_queries, list):
            raise TypeError("query expansion output must contain array field `queries`")
        return self._normalize_queries(raw_queries)[: max(0, int(max_queries))]

    def _build_rule_queries(self, *, query: str) -> list[str]:
        normalized = clean_whitespace(query)
        if not normalized:
            return []
        tokens = tokenize_for_query(normalized)
        language = self._detect_language(normalized)
        variants = self._build_rule_variants(
            normalized=normalized,
            tokens=tokens,
            language=language,
        )
        return self._normalize_queries(variants)

    def _build_rule_variants(
        self,
        *,
        normalized: str,
        tokens: list[str],
        language: str,
    ) -> list[str]:
        out: list[str] = []
        phrase_variant = self._build_phrase_variant(normalized)
        if phrase_variant:
            out.append(phrase_variant)
        compact_variant = self._build_compact_variant(tokens=tokens, language=language)
        if compact_variant:
            out.append(compact_variant)
        out.append(f"{normalized} {self._intent_suffix(language)}")
        out.append(f"{normalized} {self._evidence_suffix(language)}")
        return out

    def _build_phrase_variant(self, normalized: str) -> str:
        return f'"{normalized}"' if normalized else ""

    def _merge_query_jobs(
        self,
        *,
        primary_query: str,
        manual_queries: list[str],
        rule_queries: list[str],
        llm_queries: list[str],
        max_extra_queries: int,
    ) -> list[SearchQueryJob]:
        candidates = self._build_query_candidates(
            primary_query=primary_query,
            manual_queries=manual_queries,
            rule_queries=rule_queries,
            llm_queries=llm_queries,
        )
        jobs = self._dedupe_candidates(
            candidates=candidates,
            max_extra_queries=max_extra_queries,
        )
        if jobs:
            return jobs
        return [SearchQueryJob(query=primary_query, source="primary")]

    def _build_query_candidates(
        self,
        *,
        primary_query: str,
        manual_queries: list[str],
        rule_queries: list[str],
        llm_queries: list[str],
    ) -> list[SearchQueryCandidate]:
        candidates: list[SearchQueryCandidate] = [
            SearchQueryCandidate(query=primary_query, source="primary")
        ]
        candidates.extend(
            SearchQueryCandidate(query=item, source="manual") for item in manual_queries
        )
        candidates.extend(
            SearchQueryCandidate(query=item, source="rule") for item in rule_queries
        )
        candidates.extend(
            SearchQueryCandidate(query=item, source="llm") for item in llm_queries
        )
        return candidates

    def _dedupe_candidates(
        self,
        *,
        candidates: list[SearchQueryCandidate],
        max_extra_queries: int,
    ) -> list[SearchQueryJob]:
        limit = max(0, int(max_extra_queries))
        exact_seen: set[str] = set()
        token_signatures: set[tuple[str, ...]] = set()
        jobs: list[SearchQueryJob] = []
        additional_count = 0
        for item in candidates:
            query = clean_whitespace(item.query)
            if not query:
                continue
            key = query.casefold()
            if key in exact_seen:
                continue
            if item.source != "primary" and additional_count >= limit:
                continue
            signature = self._build_token_signature(query)
            if signature and signature in token_signatures:
                continue
            exact_seen.add(key)
            if signature:
                token_signatures.add(signature)
            jobs.append(
                SearchQueryJob(
                    query=query,
                    source=self._normalize_source(item.source),
                )
            )
            if item.source != "primary":
                additional_count += 1
        return jobs

    def _build_token_signature(self, query: str) -> tuple[str, ...]:
        tokens = sorted({token for token in tokenize_for_query(query) if token})
        return tuple(tokens)

    def _normalize_source(
        self, source: str
    ) -> Literal["primary", "manual", "rule", "llm"]:
        if source == "manual":
            return "manual"
        if source == "rule":
            return "rule"
        if source == "llm":
            return "llm"
        return "primary"

    def _normalize_queries(self, values: Sequence[object]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values:
            query = clean_whitespace(str(item or ""))
            if not query:
                continue
            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(query)
        return out

    def _detect_language(self, text: str) -> str:
        if _RE_KANA.search(text):
            return "ja"
        if _RE_CJK.search(text):
            return "zh"
        return "other"

    def _intent_suffix(self, language: str) -> str:
        if language == "zh":
            return _ZH_INTENT_SUFFIX
        if language == "ja":
            return _JA_INTENT_SUFFIX
        return _EN_INTENT_SUFFIX

    def _evidence_suffix(self, language: str) -> str:
        if language == "zh":
            return _ZH_EVIDENCE_SUFFIX
        if language == "ja":
            return _JA_EVIDENCE_SUFFIX
        return _EN_EVIDENCE_SUFFIX

    def _build_compact_variant(self, *, tokens: list[str], language: str) -> str:
        if language == "ja":
            return ""
        seen: set[str] = set()
        kept: list[str] = []
        for raw in tokens:
            token = clean_whitespace(str(raw or ""))
            if not token:
                continue
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            if language in {"zh", "ja"}:
                if len(token) < 2:
                    continue
            else:
                if self._is_version_like_token(token):
                    kept.append(token)
                    continue
                if token.isdigit() and len(token) != 4:
                    continue
                if not token.isdigit() and len(token) < 3:
                    continue
            kept.append(token)
        if len(kept) < 2:
            return ""
        compact = clean_whitespace(" ".join(kept))
        if compact.casefold() == clean_whitespace(" ".join(tokens)).casefold():
            return ""
        return compact

    def _is_version_like_token(self, token: str) -> bool:
        return bool(_RE_VERSION_LIKE_TOKEN.fullmatch(token))

    def _try_parse_json_value(self, text: str) -> object:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                return json.loads(text[start : end + 1])
            raise


__all__ = ["SearchExpandStep"]
