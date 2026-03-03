from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import override
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from serpsage.app.request import (
    FetchContentRequest,
    FetchOthersRequest,
    FetchRequestBase,
    FetchSubpagesRequest,
    SearchRequest,
)
from serpsage.app.response import FetchResultItem
from serpsage.models.pipeline import (
    ResearchSearchJob,
    ResearchSource,
    ResearchStepContext,
    SearchFetchedCandidate,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.language import (
    map_provider_language_param,
    normalize_language_code,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.steps.base import RunnerBase

_TRACKING_QUERY_KEYS = {"gclid", "fbclid", "msclkid"}
_TRACKING_QUERY_PREFIXES = ("utm_",)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_CORPUS_SCORE_WEIGHT_NEWNESS = 0.45
_CORPUS_SCORE_WEIGHT_RELEVANCE = 0.30
_CORPUS_SCORE_WEIGHT_DEPTH = 0.15
_CORPUS_SCORE_WEIGHT_STABILITY = 0.10


@dataclass(slots=True)
class CorpusUpsertResult:
    source_id: int
    canonical_url: str
    is_new_canonical: bool
    is_new_version: bool


class ResearchSearchStep(StepBase[ResearchStepContext]):
    def __init__(
        self,
        *,
        rt: Runtime,
        search_runner: RunnerBase[SearchStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._search_runner = search_runner
        self.bind_deps(search_runner)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        if ctx.runtime.stop or ctx.current_round is None:
            return ctx
        ctx.work.search_fetched_candidates = []
        round_action = clean_whitespace(
            str(ctx.work.round_action or "search")
        ).casefold()
        if int(ctx.current_round.round_index) <= 1:
            round_action = "search"
        if round_action != "search":
            return ctx
        return await self._run_search_action(ctx)

    async def _run_search_action(self, ctx: ResearchStepContext) -> ResearchStepContext:
        assert ctx.current_round is not None
        jobs = list(ctx.work.search_jobs)
        if not jobs:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "no_search_jobs"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "no_search_jobs"
            return ctx
        remaining_search_budget = max(
            0,
            int(ctx.runtime.budget.max_search_calls) - int(ctx.runtime.search_calls),
        )
        if remaining_search_budget <= 0:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_search_calls"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "max_search_calls"
            return ctx
        remaining_fetch_budget = max(
            0,
            int(ctx.runtime.budget.max_fetch_calls) - int(ctx.runtime.fetch_calls),
        )
        if remaining_fetch_budget <= 0:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_fetch_calls"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "max_fetch_calls"
            return ctx
        executable_jobs = min(
            len(jobs),
            int(remaining_fetch_budget),
            int(remaining_search_budget),
        )
        jobs = jobs[:executable_jobs]
        if not jobs:
            ctx.runtime.stop = remaining_search_budget <= 0
            ctx.runtime.stop_reason = (
                "max_search_calls"
                if remaining_search_budget <= 0
                else "max_fetch_calls"
            )
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = str(ctx.runtime.stop_reason)
            return ctx
        mode_depth = ctx.runtime.mode_depth
        main_links_limit = max(1, int(mode_depth.search_links_main_limit))
        search_language = normalize_language_code(
            ctx.plan.theme_plan.search_language,
            default="other",
        )
        if search_language == "other":
            search_language = normalize_language_code(
                ctx.plan.theme_plan.output_language
                or ctx.plan.theme_plan.input_language,
                default="en",
            )
        provider_params = map_provider_language_param(
            provider_backend=str(self.settings.provider.backend),
            search_language=search_language,
        )
        if provider_params:
            ctx.runtime.provider_language_param_applied = True
        contexts: list[SearchStepContext] = []
        for idx, job in enumerate(jobs):
            jobs_left_after_current = max(0, len(jobs) - idx - 1)
            fetch_cap = max(1, remaining_fetch_budget - jobs_left_after_current)
            max_subpages = max(0, int(fetch_cap) - 1)
            subpages_request = (
                FetchSubpagesRequest(
                    max_subpages=max_subpages,
                    subpage_keywords=job.query,
                )
                if max_subpages > 0
                else None
            )
            req = self._build_search_request(
                ctx=ctx,
                query_job=job,
                subpages_request=subpages_request,
                main_links_limit=main_links_limit,
            )
            contexts.append(
                SearchStepContext(
                    settings=ctx.settings,
                    request=req,
                    disable_internal_llm=True,
                    provider_params=dict(provider_params),
                    request_id=f"{ctx.request_id}:research:{ctx.current_round.round_index}:{idx}",
                )
            )
            remaining_fetch_budget = max(0, int(remaining_fetch_budget) - 1)
        out = await self._search_runner.run_batch(contexts)
        prepared_candidates: list[SearchFetchedCandidate] = []
        for search_ctx in out:
            results = list(search_ctx.output.results or [])
            candidates = list(search_ctx.fetch.candidates or [])
            consumed_indexes: set[int] = set()
            for result in results:
                candidate = self._match_candidate_for_result(
                    result=result,
                    candidates=candidates,
                    consumed_indexes=consumed_indexes,
                )
                if candidate is None:
                    candidate = SearchFetchedCandidate(result=result)
                else:
                    candidate = candidate.model_copy(update={"result": result})
                prepared_candidates.append(candidate.model_copy(deep=True))
        ctx.work.search_fetched_candidates = [
            item.model_copy(deep=True) for item in prepared_candidates
        ]
        ctx.runtime.search_calls += int(len(jobs))
        return ctx

    def _build_search_request(
        self,
        *,
        ctx: ResearchStepContext,
        query_job: ResearchSearchJob,
        subpages_request: FetchSubpagesRequest | None,
        main_links_limit: int,
    ) -> SearchRequest:
        return SearchRequest(
            query=str(query_job.query),
            additional_queries=(
                list(query_job.additional_queries or [])
                if str(query_job.mode) == "deep"
                else None
            ),
            mode=query_job.mode,
            max_results=int(ctx.runtime.budget.max_results_per_search),
            include_domains=(list(query_job.include_domains) or None),
            exclude_domains=(list(query_job.exclude_domains) or None),
            include_text=(list(query_job.include_text) or None),
            exclude_text=(list(query_job.exclude_text) or None),
            fetchs=FetchRequestBase(
                crawl_mode="fallback",
                crawl_timeout=30.0,
                content=FetchContentRequest(
                    detail="full",
                    include_markdown_links=False,
                    include_html_tags=False,
                ),
                abstracts=False,
                subpages=subpages_request,
                overview=True,
                others=FetchOthersRequest(max_links=int(main_links_limit)),
            ),
        )

    def _match_candidate_for_result(
        self,
        *,
        result: FetchResultItem,
        candidates: list[SearchFetchedCandidate],
        consumed_indexes: set[int],
    ) -> SearchFetchedCandidate | None:
        target_url = clean_whitespace(str(result.url))
        target_key = canonicalize_url(target_url) or target_url.casefold()
        for index, candidate in enumerate(candidates):
            if index in consumed_indexes:
                continue
            candidate_url = clean_whitespace(str(candidate.result.url))
            candidate_key = canonicalize_url(candidate_url) or candidate_url.casefold()
            if candidate_key != target_key:
                continue
            consumed_indexes.add(index)
            return candidate
        return None


def canonicalize_url(raw_url: str) -> str:
    token = clean_whitespace(raw_url)
    if not token:
        return ""
    try:
        parsed = urlsplit(token)
    except Exception:  # noqa: S112
        return token
    scheme = clean_whitespace(parsed.scheme).lower() or "https"
    host = clean_whitespace(parsed.netloc).lower()
    path = str(parsed.path or "/").strip() or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/":
        path = path.rstrip("/") or "/"
    pairs: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        norm_key = clean_whitespace(key)
        if not norm_key:
            continue
        key_lc = norm_key.casefold()
        if key_lc in _TRACKING_QUERY_KEYS:
            continue
        if any(key_lc.startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES):
            continue
        pairs.append((norm_key, clean_whitespace(value)))
    pairs.sort(key=lambda item: (item[0].casefold(), item[1]))
    query = urlencode(pairs, doseq=True)
    return urlunsplit((scheme, host, path, query, ""))


def append_source_version(
    *,
    ctx: ResearchStepContext,
    url: str,
    title: str,
    overview: object | None,
    content: str,
    round_index: int,
    is_subpage: bool,
) -> CorpusUpsertResult:
    canonical_url = canonicalize_url(url) or clean_whitespace(url)
    normalized_title = clean_whitespace(title)
    normalized_overview = _normalize_overview(overview)
    normalized_content = _normalize_text(content)
    fingerprint = build_content_fingerprint(
        content=normalized_content,
        overview=normalized_overview,
    )
    synchronize_corpus_indexes(ctx=ctx)
    source_idx_by_id = _build_source_idx_by_id(ctx.corpus.sources)
    existing_ids = list(ctx.corpus.source_url_to_ids.get(canonical_url, []))
    for source_id in reversed(existing_ids):
        idx = source_idx_by_id.get(source_id)
        if idx is None:
            continue
        source = ctx.corpus.sources[idx]
        if int(source.round_index) != int(round_index):
            continue
        if clean_whitespace(source.content_fingerprint) != fingerprint:
            continue
        updated = source.model_copy(
            update={
                "title": clean_whitespace(source.title) or normalized_title,
                "overview": source.overview or normalized_overview,
                "content": source.content or normalized_content,
                "seen_count": max(1, int(source.seen_count)) + 1,
                "canonical_url": canonical_url,
                "content_fingerprint": fingerprint,
            }
        )
        ctx.corpus.sources[idx] = updated
        return CorpusUpsertResult(
            source_id=int(source_id),
            canonical_url=canonical_url,
            is_new_canonical=False,
            is_new_version=False,
        )
    source_id = len(ctx.corpus.sources) + 1
    ctx.corpus.sources.append(
        ResearchSource(
            source_id=source_id,
            url=clean_whitespace(url),
            canonical_url=canonical_url,
            title=normalized_title,
            overview=normalized_overview,
            content=normalized_content,
            round_index=int(round_index),
            is_subpage=bool(is_subpage),
            seen_count=1,
            content_fingerprint=fingerprint,
        )
    )
    ids = list(ctx.corpus.source_url_to_ids.get(canonical_url, []))
    ids.append(int(source_id))
    ctx.corpus.source_url_to_ids[canonical_url] = ids
    return CorpusUpsertResult(
        source_id=int(source_id),
        canonical_url=canonical_url,
        is_new_canonical=bool(len(existing_ids) == 0),
        is_new_version=True,
    )


def rebuild_corpus_ranking(
    *,
    ctx: ResearchStepContext,
    round_index: int,
) -> float:
    synchronize_corpus_indexes(ctx=ctx)
    old_scores = dict(ctx.corpus.source_scores)
    old_ranked = list(ctx.corpus.ranked_source_ids)
    source_idx_by_id = _build_source_idx_by_id(ctx.corpus.sources)
    canonical_to_latest: dict[str, int] = {}
    canonical_seen: dict[str, int] = {}
    for source in ctx.corpus.sources:
        canonical = clean_whitespace(source.canonical_url)
        if not canonical:
            canonical = canonicalize_url(source.url) or clean_whitespace(source.url)
        if not canonical:
            continue
        canonical_seen[canonical] = canonical_seen.get(canonical, 0) + max(
            1, int(source.seen_count)
        )
        prev = canonical_to_latest.get(canonical)
        if prev is None:
            canonical_to_latest[canonical] = int(source.source_id)
            continue
        prev_source = ctx.corpus.sources[source_idx_by_id[prev]]
        if (
            int(source.round_index),
            int(source.source_id),
        ) >= (
            int(prev_source.round_index),
            int(prev_source.source_id),
        ):
            canonical_to_latest[canonical] = int(source.source_id)
    max_seen = max(canonical_seen.values(), default=1)
    query_tokens = _build_query_tokens(ctx=ctx)
    scored: list[tuple[int, float]] = []
    score_map: dict[int, float] = {}
    for canonical, latest_id in canonical_to_latest.items():
        idx = source_idx_by_id.get(latest_id)
        if idx is None:
            continue
        source = ctx.corpus.sources[idx]
        newness_score = _compute_newness_score(
            source_round_index=int(source.round_index),
            current_round_index=int(round_index),
        )
        relevance_score = _compute_relevance_score(
            source=source,
            query_tokens=query_tokens,
        )
        depth_score = _compute_depth_score(source)
        stability_score = min(
            1.0,
            float(canonical_seen.get(canonical, 0)) / float(max_seen or 1),
        )
        final_score = (
            float(_CORPUS_SCORE_WEIGHT_NEWNESS) * newness_score
            + float(_CORPUS_SCORE_WEIGHT_RELEVANCE) * relevance_score
            + float(_CORPUS_SCORE_WEIGHT_DEPTH) * depth_score
            + float(_CORPUS_SCORE_WEIGHT_STABILITY) * stability_score
        )
        score_map[int(latest_id)] = float(final_score)
        scored.append((int(latest_id), float(final_score)))
    scored.sort(
        key=lambda item: (
            float(item[1]),
            _source_round_index(ctx=ctx, source_id=item[0]),
            int(item[0]),
        ),
        reverse=True,
    )
    ranked_source_ids = [source_id for source_id, _ in scored]
    full_score_map: dict[int, float] = {}
    for canonical, ids in ctx.corpus.source_url_to_ids.items():
        canonical_score = 0.0
        latest_source_id = canonical_to_latest.get(canonical)
        if latest_source_id is not None:
            canonical_score = float(score_map.get(int(latest_source_id), 0.0))
        for source_id in ids:
            full_score_map[int(source_id)] = canonical_score
    for idx, source in enumerate(ctx.corpus.sources):
        canonical = clean_whitespace(source.canonical_url) or canonicalize_url(
            source.url
        )
        if canonical != source.canonical_url:
            ctx.corpus.sources[idx] = source.model_copy(
                update={
                    "canonical_url": canonical,
                }
            )
    ctx.corpus.ranked_source_ids = list(ranked_source_ids)
    ctx.corpus.source_scores = dict(full_score_map)
    old_total = sum(
        float(old_scores.get(int(source_id), 0.0)) for source_id in old_ranked
    )
    new_total = sum(
        float(ctx.corpus.source_scores.get(int(source_id), 0.0))
        for source_id in ranked_source_ids
    )
    return max(0.0, float(new_total - old_total))


def select_context_source_ids(
    *,
    ctx: ResearchStepContext,
    round_index: int,
    topk: int,
    new_result_target_ratio: float,
    min_history_sources: int,
) -> list[int]:
    limit = max(1, int(topk))
    ranked_ids = _resolve_ranked_source_ids(ctx=ctx)
    if not ranked_ids:
        return []
    source_by_id = {int(source.source_id): source for source in ctx.corpus.sources}
    new_ids = [
        source_id
        for source_id in ranked_ids
        if int(source_by_id[source_id].round_index) == int(round_index)
    ]
    history_ids = [
        source_id
        for source_id in ranked_ids
        if int(source_by_id[source_id].round_index) != int(round_index)
    ]
    target_new = min(
        len(new_ids),
        int(
            math.ceil(float(limit) * float(max(0.0, min(1.0, new_result_target_ratio))))
        ),
    )
    selected: list[int] = []
    selected.extend(new_ids[:target_new])
    selected.extend(history_ids[: max(0, limit - len(selected))])
    if len(selected) < limit:
        for source_id in new_ids[target_new:]:
            if source_id in selected:
                continue
            selected.append(source_id)
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        for source_id in history_ids:
            if source_id in selected:
                continue
            selected.append(source_id)
            if len(selected) >= limit:
                break
    min_history = max(0, int(min_history_sources))
    history_needed = min(min_history, len(history_ids))
    history_selected = sum(1 for source_id in selected if source_id in history_ids)
    if history_selected < history_needed:
        for source_id in history_ids:
            if source_id in selected:
                continue
            selected.append(source_id)
            history_selected += 1
            if history_selected >= history_needed:
                break
    if len(selected) > limit:
        selected = _trim_to_limit(
            selected=selected,
            source_by_id=source_by_id,
            round_index=round_index,
            limit=limit,
        )
    rank_index = {source_id: idx for idx, source_id in enumerate(ranked_ids)}
    deduped: list[int] = []
    seen: set[int] = set()
    for source_id in selected:
        if source_id in seen:
            continue
        seen.add(source_id)
        deduped.append(source_id)
    deduped.sort(key=lambda source_id: rank_index.get(source_id, 10**9))
    return deduped[:limit]


def pick_sources_by_ids(
    *,
    sources: list[ResearchSource],
    source_ids: list[int],
) -> list[ResearchSource]:
    source_by_id = {int(source.source_id): source for source in sources}
    out: list[ResearchSource] = []
    for source_id in source_ids:
        source = source_by_id.get(int(source_id))
        if source is None:
            continue
        out.append(source)
    return out


def sort_source_ids_by_score(
    *,
    ctx: ResearchStepContext,
    source_ids: list[int],
) -> list[int]:
    source_by_id = {int(source.source_id): source for source in ctx.corpus.sources}
    out: list[int] = []
    seen: set[int] = set()
    for raw in source_ids:
        source_id = int(raw)
        if source_id in seen or source_id not in source_by_id:
            continue
        seen.add(source_id)
        out.append(source_id)
    out.sort(
        key=lambda source_id: (
            float(ctx.corpus.source_scores.get(source_id, 0.0)),
            int(source_by_id[source_id].round_index),
            int(source_id),
        ),
        reverse=True,
    )
    return out


def synchronize_corpus_indexes(*, ctx: ResearchStepContext) -> None:
    sorted_sources = sorted(ctx.corpus.sources, key=lambda item: int(item.source_id))
    url_to_ids: dict[str, list[int]] = {}
    for idx, source in enumerate(sorted_sources):
        canonical = clean_whitespace(source.canonical_url) or canonicalize_url(
            source.url
        )
        if canonical != source.canonical_url:
            source = source.model_copy(update={"canonical_url": canonical})
            sorted_sources[idx] = source
        if not canonical:
            continue
        ids = list(url_to_ids.get(canonical, []))
        ids.append(int(source.source_id))
        url_to_ids[canonical] = ids
    ctx.corpus.sources = sorted_sources
    ctx.corpus.source_url_to_ids = url_to_ids


def build_content_fingerprint(*, content: str, overview: object | None) -> str:
    normalized_content = _normalize_text(content)
    lines = [normalized_content[:5000]]
    overview_text = _normalize_overview_text(overview)
    if overview_text:
        lines.append(overview_text[:5000])
    payload = "\n".join(lines).strip() or "__empty__"
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def _resolve_ranked_source_ids(*, ctx: ResearchStepContext) -> list[int]:
    source_ids: list[int] = []
    source_by_id = {int(source.source_id): source for source in ctx.corpus.sources}
    seen_canonical: set[str] = set()
    for source_id in ctx.corpus.ranked_source_ids:
        source = source_by_id.get(int(source_id))
        if source is None:
            continue
        canonical = clean_whitespace(source.canonical_url) or canonicalize_url(
            source.url
        )
        if not canonical or canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        source_ids.append(int(source_id))
    if source_ids:
        return source_ids
    fallback = sorted(
        ctx.corpus.sources,
        key=lambda item: (int(item.round_index), int(item.source_id)),
        reverse=True,
    )
    out: list[int] = []
    for source in fallback:
        canonical = clean_whitespace(source.canonical_url) or canonicalize_url(
            source.url
        )
        if not canonical or canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        out.append(int(source.source_id))
    return out


def _trim_to_limit(
    *,
    selected: list[int],
    source_by_id: dict[int, ResearchSource],
    round_index: int,
    limit: int,
) -> list[int]:
    out = list(selected)
    while len(out) > limit:
        removed = False
        for idx in range(len(out) - 1, -1, -1):
            source_id = out[idx]
            if int(source_by_id[source_id].round_index) == int(round_index):
                out.pop(idx)
                removed = True
                break
        if not removed:
            out.pop()
    return out


def _source_round_index(*, ctx: ResearchStepContext, source_id: int) -> int:
    for source in ctx.corpus.sources:
        if int(source.source_id) == int(source_id):
            return int(source.round_index)
    return 0


def _compute_newness_score(
    *,
    source_round_index: int,
    current_round_index: int,
) -> float:
    round_gap = max(0, int(current_round_index) - int(source_round_index))
    if round_gap <= 0:
        return 1.0
    return max(0.0, 1.0 - (float(round_gap) / 5.0))


def _compute_relevance_score(
    *,
    source: ResearchSource,
    query_tokens: set[str],
) -> float:
    if not query_tokens:
        return 0.0
    parts = [
        clean_whitespace(source.title),
        clean_whitespace(source.url),
        _normalize_overview_text(source.overview)[:1800],
        _normalize_text(source.content)[:1800],
    ]
    haystack = " ".join(parts).casefold()
    if not haystack:
        return 0.0
    hits = sum(1 for token in query_tokens if token in haystack)
    return min(1.0, float(hits) / float(len(query_tokens)))


def _compute_depth_score(source: ResearchSource) -> float:
    content_len = len(_normalize_text(source.content))
    content_score = min(1.0, float(content_len) / 2400.0)
    overview_len = len(_normalize_overview_text(source.overview))
    overview_score = min(1.0, float(overview_len) / 1200.0)
    return min(1.0, 0.7 * content_score + 0.3 * overview_score)


def _build_query_tokens(*, ctx: ResearchStepContext) -> set[str]:
    tokens: set[str] = set()
    values = list(ctx.plan.next_queries)
    if ctx.current_round is not None:
        values.extend(ctx.current_round.queries)
    values.append(_resolve_core_question(ctx=ctx))
    for value in values:
        for token in _TOKEN_PATTERN.findall(str(value).casefold()):
            if len(token) < 2:
                continue
            tokens.add(token)
    return tokens


def _build_source_idx_by_id(sources: list[ResearchSource]) -> dict[int, int]:
    return {int(item.source_id): idx for idx, item in enumerate(sources)}


def _resolve_core_question(*, ctx: ResearchStepContext) -> str:
    question = clean_whitespace(ctx.plan.theme_plan.core_question or ctx.request.themes)
    return question or clean_whitespace(ctx.request.themes)


def _normalize_text(raw: object) -> str:
    return str(raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_overview(raw: object | None) -> str | object | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        token = _normalize_text(raw)
        return token or None
    return raw


def _normalize_overview_text(raw: object | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return _normalize_text(raw)
    try:
        return _normalize_text(json.dumps(raw, ensure_ascii=False, sort_keys=True))
    except Exception:  # noqa: S112
        return _normalize_text(str(raw))


__all__ = [
    "CorpusUpsertResult",
    "ResearchSearchStep",
    "append_source_version",
    "build_content_fingerprint",
    "canonicalize_url",
    "pick_sources_by_ids",
    "rebuild_corpus_ranking",
    "select_context_source_ids",
    "sort_source_ids_by_score",
    "synchronize_corpus_indexes",
]
