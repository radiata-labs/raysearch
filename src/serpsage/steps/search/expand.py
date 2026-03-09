from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.core.runtime import Runtime
from serpsage.dependencies import Inject
from serpsage.models.steps.search import (
    SearchDeepState,
    SearchQueryCandidate,
    SearchQueryJob,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from collections.abc import Sequence
_JACCARD_SIMILARITY_THRESHOLD = 0.92
_MANUAL_SOURCE_CAP = 8
_RULE_SOURCE_CAP = 8
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
    def __init__(
        self, *, rt: Runtime = Inject(), llm: LLMClientBase = Inject()
    ) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        """Build deep-search query jobs from primary/manual/rule/LLM variants.
        Args:
            ctx: Search pipeline context containing request and deep state.
        Returns:
            Updated context with `deep.query_jobs` populated for deep mode.
        """
        ctx.deep = SearchDeepState()
        req = ctx.request
        if not self._is_deep_enabled(req_mode=str(req.mode or "auto")):
            return ctx
        primary_query = clean_whitespace(str(req.query or ""))
        if not primary_query:
            await self._abort_empty_query(ctx=ctx, raw_query=req.query)
            return ctx
        deep_cfg = self.settings.search.deep
        manual_queries = self._get_manual_queries(
            req_additional_queries=req.additional_queries
        )
        disable_internal_llm = bool(ctx.disable_internal_llm)
        if disable_internal_llm:
            llm_queries: list[str] = []
        else:
            llm_queries = await self._collect_llm_queries(
                ctx=ctx,
                query=primary_query,
                max_queries=int(deep_cfg.llm_max_queries),
                timeout_s=float(deep_cfg.expansion_timeout_s),
            )
        rule_queries = self._build_rule_queries(
            query=primary_query,
            max_queries=int(deep_cfg.rule_max_queries),
        )
        query_jobs = self._merge_query_jobs(
            primary_query=primary_query,
            manual_queries=manual_queries,
            rule_queries=rule_queries,
            llm_queries=llm_queries,
            max_expanded_queries=int(deep_cfg.max_expanded_queries),
            manual_weight=float(deep_cfg.manual_query_score_weight),
            rule_weight=float(deep_cfg.rule_query_score_weight),
            llm_weight=float(deep_cfg.llm_query_score_weight),
        )
        ctx.deep.query_jobs = query_jobs
        return ctx

    def _is_deep_enabled(self, *, req_mode: str) -> bool:
        return req_mode == "deep" and bool(self.settings.search.deep.enabled)

    def _get_manual_queries(
        self, *, req_additional_queries: list[str] | None
    ) -> list[str]:
        return self._normalize_queries(list(req_additional_queries or []))

    async def _abort_empty_query(
        self, *, ctx: SearchStepContext, raw_query: object
    ) -> None:
        ctx.deep.aborted = True
        ctx.deep.abort_reason = "empty query"
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
        timeout_s: float,
    ) -> list[str]:
        llm_limit = max(0, int(max_queries))
        if llm_limit <= 0:
            return []
        model_name = self._resolve_expansion_model()
        try:
            return await self._expand_with_llm(
                query=query,
                model_name=model_name,
                max_queries=llm_limit,
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
            str(self.settings.search.deep.expansion_model or "")
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
        result = await self._llm.create(
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

    def _build_rule_queries(self, *, query: str, max_queries: int) -> list[str]:
        if max_queries <= 0:
            return []
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
        return self._normalize_queries(variants)[:max_queries]

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
        max_expanded_queries: int,
        manual_weight: float,
        rule_weight: float,
        llm_weight: float,
    ) -> list[SearchQueryJob]:
        candidates = self._build_query_candidates(
            primary_query=primary_query,
            manual_queries=manual_queries,
            rule_queries=rule_queries,
            llm_queries=llm_queries,
            manual_weight=manual_weight,
            rule_weight=rule_weight,
            llm_weight=llm_weight,
        )
        jobs = self._dedupe_candidates(
            candidates=candidates,
            max_expanded_queries=max_expanded_queries,
        )
        if jobs:
            return jobs
        return [SearchQueryJob(query=primary_query, weight=1.0, source="primary")]

    def _build_query_candidates(
        self,
        *,
        primary_query: str,
        manual_queries: list[str],
        rule_queries: list[str],
        llm_queries: list[str],
        manual_weight: float,
        rule_weight: float,
        llm_weight: float,
    ) -> list[SearchQueryCandidate]:
        candidates: list[SearchQueryCandidate] = [
            SearchQueryCandidate(query=primary_query, weight=1.0, source="primary")
        ]
        candidates.extend(
            SearchQueryCandidate(query=item, weight=manual_weight, source="manual")
            for item in manual_queries
        )
        candidates.extend(
            SearchQueryCandidate(query=item, weight=rule_weight, source="rule")
            for item in rule_queries
        )
        candidates.extend(
            SearchQueryCandidate(query=item, weight=llm_weight, source="llm")
            for item in llm_queries
        )
        return candidates

    def _dedupe_candidates(
        self,
        *,
        candidates: list[SearchQueryCandidate],
        max_expanded_queries: int,
    ) -> list[SearchQueryJob]:
        limit = max(0, int(max_expanded_queries))
        exact_seen: set[str] = set()
        kept_token_sets: list[set[str]] = []
        jobs: list[SearchQueryJob] = []
        additional_count = 0
        source_counts: dict[str, int] = {"manual": 0, "rule": 0, "llm": 0}
        for item in candidates:
            query = clean_whitespace(item.query)
            if not query:
                continue
            key = query.casefold()
            if key in exact_seen:
                continue
            if item.source != "primary" and additional_count >= limit:
                continue
            if (
                item.source == "manual"
                and source_counts["manual"] >= _MANUAL_SOURCE_CAP
            ):
                continue
            if item.source == "rule" and source_counts["rule"] >= _RULE_SOURCE_CAP:
                continue
            token_set = set(tokenize_for_query(query))
            if self._is_near_duplicate(token_set, kept_token_sets):
                continue
            exact_seen.add(key)
            kept_token_sets.append(token_set)
            jobs.append(
                SearchQueryJob(
                    query=query,
                    weight=float(max(0.0, item.weight)),
                    source=self._normalize_source(item.source),
                )
            )
            if item.source != "primary":
                additional_count += 1
                if item.source in source_counts:
                    source_counts[item.source] += 1
        return jobs

    def _is_near_duplicate(
        self, token_set: set[str], kept_token_sets: list[set[str]]
    ) -> bool:
        if not token_set:
            return False
        return any(
            self._jaccard_similarity(token_set, prev) >= _JACCARD_SIMILARITY_THRESHOLD
            for prev in kept_token_sets
            if prev
        )

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

    def _jaccard_similarity(self, left: set[str], right: set[str]) -> float:
        union_size = len(left | right)
        if union_size <= 0:
            return 1.0
        return float(len(left & right) / union_size)

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
            # Sudachi/jieba mixed tokenization can split Japanese words unnaturally.
            # Skip compact rewrite for Japanese to avoid malformed queries.
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
                    if len(kept) >= 4:
                        break
                    continue
                if token.isdigit() and len(token) != 4:
                    continue
                if not token.isdigit() and len(token) < 3:
                    continue
            kept.append(token)
            if len(kept) >= 4:
                break
        if len(kept) < 2:
            return ""
        return " ".join(kept)

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
