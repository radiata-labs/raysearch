from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Literal
from typing_extensions import override
from urllib.parse import urljoin, urlsplit, urlunsplit

import anyio

from raysearch.components.llm.base import LLMClientBase
from raysearch.components.rank.base import RankerBase
from raysearch.dependencies import FETCH_RUNNER, Depends
from raysearch.models.app.request import (
    FetchContentRequest,
    FetchOthersRequest,
    FetchRequest,
)
from raysearch.models.app.response import (
    FetchResponse,
    FetchResultItem,
    FetchSubpagesResult,
)
from raysearch.models.components.extract import ExtractRef
from raysearch.models.steps.fetch import FetchStepContext
from raysearch.models.steps.research import (
    ResearchCorpusUpsertResult,
    ResearchSource,
    RoundStepContext,
)
from raysearch.models.steps.research.payloads import ResearchLinkPickerPayload
from raysearch.models.steps.search import QuerySourceSpec, SearchFetchedCandidate
from raysearch.steps.base import RunnerBase, StepBase
from raysearch.steps.research.prompt import build_link_picker_prompt_messages
from raysearch.steps.research.schema import build_link_picker_schema
from raysearch.steps.research.utils import (
    canonicalize_url,
    resolve_research_model,
    source_authority_score,
)
from raysearch.tokenize import tokenize_for_query
from raysearch.utils import clean_whitespace

_EXPLORE_ALLOWED_SCHEMES = {"http", "https"}
_EXPLORE_LOW_VALUE_PATH_PREFIXES = (
    "privacy",
    "terms",
    "login",
    "signin",
    "signup",
    "logout",
)
_LINK_PICKER_PRERANK_MULTIPLIER = 4
_LINK_PICKER_PRERANK_FLOOR = 16

# Corpus scoring weights
_CORPUS_SCORE_WEIGHT_NEWNESS = 0.35
_CORPUS_SCORE_WEIGHT_RELEVANCE = 0.25
_CORPUS_SCORE_WEIGHT_DEPTH = 0.15
_CORPUS_SCORE_WEIGHT_STABILITY = 0.10
_CORPUS_SCORE_WEIGHT_AUTHORITY = 0.15

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


class ResearchFetchStep(StepBase[RoundStepContext]):
    fetch_runner: RunnerBase[FetchStepContext] = Depends(FETCH_RUNNER)
    ranker: RankerBase = Depends()
    llm: LLMClientBase = Depends()

    def __init__(
        self,
        *,
        phase: Literal["pre", "post"] = "post",
    ) -> None:
        phase_key = phase.casefold()
        if phase_key not in {"pre", "post"}:
            phase_key = "post"
        self.phase: Literal["pre", "post"] = "pre" if phase_key == "pre" else "post"

    @override
    async def run_inner(self, ctx: RoundStepContext) -> RoundStepContext:
        if ctx.run.stop or ctx.run.current is None:
            return ctx
        round_action = (ctx.run.current.round_action or "search").casefold()
        if ctx.run.current.round_index <= 1:
            round_action = "search"
        if self.phase == "pre":
            if round_action == "explore":
                executed = await self._run_explore_fetch(ctx)
                if executed:
                    ctx.run.current.search_fetched_candidates = []
                    return ctx
                ctx.run.current.round_action = "search"
                ctx.run.current.explore_target_source_ids = []
            return ctx
        if round_action != "search":
            return ctx
        ctx = self._run_search_fetch(ctx)
        if (
            ctx.run.current is not None
            and not ctx.run.current.waiting_for_budget
            and not ctx.run.current.pending_search_jobs
        ):
            ctx.run.current.search_fetched_candidates = []
        return ctx

    def _run_search_fetch(self, ctx: RoundStepContext) -> RoundStepContext:
        assert ctx.run.current is not None
        candidates = [
            item.model_copy(deep=True)
            for item in list(ctx.run.current.search_fetched_candidates)
        ]
        remaining_fetch_budget = ctx.run.allocation.fetch_remaining
        if remaining_fetch_budget <= 0:
            ctx.run.current.waiting_for_budget = True
            ctx.run.current.waiting_reason = "max_fetch_calls"
            return ctx
        if ctx.run.current.pending_search_jobs:
            return ctx
        all_results: list[FetchResultItem] = []
        per_round_fetch_calls = 0
        round_link_candidates: dict[int, SearchFetchedCandidate] = {}
        selected_candidates, deferred_candidates, per_round_fetch_calls = (
            self._select_candidates_for_fetch_budget(
                candidates=candidates,
                remaining_fetch_budget=remaining_fetch_budget,
            )
        )
        for trimmed_candidate in selected_candidates:
            all_results.append(trimmed_candidate.result)
            main_source_id, _, _ = self._append_sources_from_fetch_result(
                ctx=ctx,
                result=trimmed_candidate.result,
                round_index=ctx.run.current.round_index,
            )
            round_link_candidates[main_source_id] = (
                self._build_link_candidate_from_search_candidate(
                    candidate=trimmed_candidate,
                )
            )
        ctx.run.current.search_fetched_candidates = [
            item.model_copy(deep=True) for item in deferred_candidates
        ]
        ctx.run.current.waiting_for_budget = bool(deferred_candidates)
        ctx.run.current.waiting_reason = (
            "max_fetch_calls" if deferred_candidates else ""
        )
        self._finalize_round_state(
            ctx=ctx,
            result_count=len(all_results),
            per_round_fetch_calls=per_round_fetch_calls,
            next_round_link_candidates=round_link_candidates,
        )
        return ctx

    async def _run_explore_fetch(self, ctx: RoundStepContext) -> bool:
        assert ctx.run.current is not None
        remaining_fetch_budget = ctx.run.allocation.fetch_remaining
        if remaining_fetch_budget <= 0:
            ctx.run.stop = True
            ctx.run.stop_reason = "max_fetch_calls"
            return False
        expected_candidate_round = ctx.run.current.round_index - 1
        if ctx.run.link_candidates_round != expected_candidate_round:
            return False
        target_pages = self._select_explore_target_pages(ctx=ctx)
        if not target_pages:
            return False
        selected_urls = await self._select_links_for_target_pages(
            ctx=ctx,
            target_pages=target_pages,
        )
        if not selected_urls:
            return False
        selected_urls = self._filter_already_fetched_urls(
            ctx=ctx,
            urls=selected_urls,
        )
        if not selected_urls:
            return False
        fetch_cap = remaining_fetch_budget
        if fetch_cap <= 0:
            return False
        selected_urls = selected_urls[:fetch_cap]
        if not selected_urls:
            return False
        fetch_contexts = self._build_explore_fetch_contexts(
            ctx=ctx,
            urls=selected_urls,
        )
        fetched = await self.fetch_runner.run_batch(fetch_contexts)
        all_results: list[FetchResultItem] = []
        per_round_fetch_calls = 0
        next_round_link_candidates: dict[int, SearchFetchedCandidate] = {}
        for item in fetched:
            if item.error.failed or item.result is None:
                continue
            result = item.result
            per_round_fetch_calls += 1
            all_results.append(result)
            main_source_id, _, _ = self._append_sources_from_fetch_result(
                ctx=ctx,
                result=result,
                round_index=ctx.run.current.round_index,
            )
            next_round_link_candidates[main_source_id] = (
                self._build_link_candidate_from_explore_result(
                    result=result,
                )
            )
        if per_round_fetch_calls <= 0:
            return False
        if not next_round_link_candidates:
            next_round_link_candidates = {
                source_id: item.model_copy(deep=True)
                for source_id, item in target_pages.items()
            }
        self._finalize_round_state(
            ctx=ctx,
            result_count=len(all_results),
            per_round_fetch_calls=per_round_fetch_calls,
            next_round_link_candidates=next_round_link_candidates,
        )
        return True

    def _finalize_round_state(
        self,
        *,
        ctx: RoundStepContext,
        result_count: int,
        per_round_fetch_calls: int,
        next_round_link_candidates: dict[int, SearchFetchedCandidate],
    ) -> None:
        assert ctx.run.current is not None
        ctx.run.link_candidates = {
            source_id: item.model_copy(deep=True)
            for source_id, item in next_round_link_candidates.items()
        }
        ctx.run.link_candidates_round = ctx.run.current.round_index
        ctx.run.current.result_count = max(
            0, ctx.run.current.result_count + max(0, result_count)
        )
        if ctx.run.current.result_count > 0:
            self._rebuild_corpus_ranking(
                ctx=ctx,
                round_index=ctx.run.current.round_index,
            )
        ctx.run.allocation.fetch_used += max(0, per_round_fetch_calls)

    def _select_explore_target_pages(
        self,
        *,
        ctx: RoundStepContext,
    ) -> dict[int, SearchFetchedCandidate]:
        if ctx.run.current is None:
            return {}
        candidates = dict(ctx.run.link_candidates or {})
        if not candidates:
            return {}
        target_ids = list(ctx.run.current.explore_target_source_ids or [])
        if not target_ids:
            return {}
        cap = max(1, ctx.run.limits.explore_target_pages_per_round)
        out: dict[int, SearchFetchedCandidate] = {}
        for source_id in target_ids:
            source = candidates.get(source_id)
            if source is None:
                continue
            out[source_id] = source.model_copy(deep=True)
            if len(out) >= cap:
                break
        return out

    async def _select_links_for_target_pages(
        self,
        *,
        ctx: RoundStepContext,
        target_pages: dict[int, SearchFetchedCandidate],
    ) -> list[str]:
        per_page_limit = max(1, ctx.run.limits.explore_links_per_page)
        if ctx.run.limits.mode_key == "research-fast":
            selected: list[str] = []
            for page in target_pages.values():
                selected.extend(
                    await self._select_links_fast(
                        ctx=ctx,
                        candidate=page,
                        max_links=per_page_limit,
                    )
                )
            return self._merge_and_dedupe_urls(selected)
        page_items = list(target_pages.items())
        selected_by_page: list[list[str] | None] = [None] * len(page_items)

        async def _pick_for_page(
            index: int,
            source_id: int,
            page: SearchFetchedCandidate,
        ) -> None:
            selected_by_page[index] = await self._select_links_with_llm(
                ctx=ctx,
                source_id=source_id,
                candidate=page,
                max_links=per_page_limit,
            )

        async with anyio.create_task_group() as tg:
            for idx, (source_id, page) in enumerate(page_items):
                tg.start_soon(_pick_for_page, idx, source_id, page)
        merged: list[str] = []
        for links in selected_by_page:
            if not links:
                continue
            merged.extend(links)
        return self._merge_and_dedupe_urls(merged)

    async def _select_links_fast(
        self,
        *,
        ctx: RoundStepContext,
        candidate: SearchFetchedCandidate,
        max_links: int,
    ) -> list[str]:
        links = self._merge_and_dedupe_links(ctx=ctx, candidate=candidate)
        if not links:
            return []
        ranked_indexes = await self._rank_link_indexes(ctx=ctx, links=links)
        out: list[str] = []
        for idx in ranked_indexes[: max(1, max_links)]:
            url = links[idx].url
            if not url:
                continue
            out.append(url)
        return out

    async def _select_links_with_llm(
        self,
        *,
        ctx: RoundStepContext,
        source_id: int,
        candidate: SearchFetchedCandidate,
        max_links: int,
    ) -> list[str]:
        links = self._merge_and_dedupe_links(ctx=ctx, candidate=candidate)
        if not links:
            return []
        links = await self._prerank_links_for_llm(
            ctx=ctx,
            links=links,
            max_links=max_links,
        )
        fallback_model = resolve_research_model(
            settings=self.settings,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        model = resolve_research_model(
            settings=self.settings,
            stage="link_select",
            fallback=fallback_model,
        )
        report_style = ctx.task.style
        if report_style not in {"decision", "explainer", "execution"}:
            report_style = "explainer"
        candidate_links_markdown = self._render_link_candidates_for_picker(links)
        selected_ids: list[int] = []
        try:
            chat_result = await self.llm.create(
                model=model,
                messages=build_link_picker_prompt_messages(
                    core_question=ctx.task.question,
                    report_style=report_style,
                    mode_depth_profile=ctx.run.limits.mode_key,
                    current_utc_date=datetime.fromtimestamp(
                        self.clock.now_ms() / 1000,
                        tz=UTC,
                    )
                    .date()
                    .isoformat(),
                    source_id=source_id,
                    source_url=candidate.result.url,
                    source_title=candidate.result.title,
                    max_links_to_select=max_links,
                    candidate_links_markdown=candidate_links_markdown,
                ),
                response_format=ResearchLinkPickerPayload,
                format_override=build_link_picker_schema(),
                retries=self.settings.research.llm_self_heal_retries,
            )
            await self.meter.record(
                name="llm.tokens",
                request_id=ctx.request_id,
                model=str(model),
                unit="token",
                tokens={
                    "prompt_tokens": int(chat_result.usage.prompt_tokens),
                    "completion_tokens": int(chat_result.usage.completion_tokens),
                    "total_tokens": int(chat_result.usage.total_tokens),
                },
            )
            selected_ids = list(chat_result.data.selected_link_ids)
        except Exception:
            selected_ids = []
        if not selected_ids:
            return await self._select_links_fast(
                ctx=ctx,
                candidate=candidate,
                max_links=max_links,
            )
        out: list[str] = []
        seen: set[str] = set()
        for link_id in selected_ids:
            if link_id <= 0 or link_id > len(links):
                continue
            url = links[link_id - 1].url
            key = canonicalize_url(url) or url.casefold()
            if not url or key in seen:
                continue
            seen.add(key)
            out.append(url)
            if len(out) >= max(1, max_links):
                break
        if out:
            return out
        return await self._select_links_fast(
            ctx=ctx,
            candidate=candidate,
            max_links=max_links,
        )

    async def _prerank_links_for_llm(
        self,
        *,
        ctx: RoundStepContext,
        links: list[ExtractRef],
        max_links: int,
    ) -> list[ExtractRef]:
        if not links:
            return []
        cap = max(
            _LINK_PICKER_PRERANK_FLOOR,
            max(1, max_links) * _LINK_PICKER_PRERANK_MULTIPLIER,
        )
        if len(links) <= cap:
            return links
        ranked_indexes = await self._rank_link_indexes(ctx=ctx, links=links)
        return [
            links[idx].model_copy(deep=True)
            for idx in ranked_indexes[: max(1, cap)]
            if idx < len(links)
        ]

    async def _rank_link_indexes(
        self,
        *,
        ctx: RoundStepContext,
        links: list[ExtractRef],
    ) -> list[int]:
        if not links:
            return []
        query = ctx.task.question
        texts = [self._render_rank_text(item) for item in links]
        try:
            scores = await self.ranker.score_texts(
                texts,
                query=query,
                query_tokens=tokenize_for_query(query),
            )
        except Exception:
            return list(range(len(links)))
        return sorted(
            range(len(links)),
            key=lambda idx: (-_score_at(scores=scores, idx=idx), idx),
        )

    def _render_rank_text(self, item: ExtractRef) -> str:
        return f"text={item.text or '(none)'}; url={item.url}"

    def _render_link_candidates_for_picker(self, links: list[ExtractRef]) -> str:
        if not links:
            return "- (none)"
        lines: list[str] = []
        for index, item in enumerate(links, start=1):
            lines.append(
                f"- id={index} | text={item.text or '(none)'} | url={item.url or 'n/a'}"
            )
        return "\n".join(lines).strip()

    def _merge_and_dedupe_links(
        self,
        *,
        ctx: RoundStepContext | None = None,
        candidate: SearchFetchedCandidate,
    ) -> list[ExtractRef]:
        merged: list[tuple[ExtractRef, str]] = [
            (item, candidate.result.url) for item in list(candidate.links or [])
        ]
        for index, links in enumerate(list(candidate.subpage_links or [])):
            base_url = (
                candidate.result.subpages[index].url
                if index < len(candidate.result.subpages)
                else candidate.result.url
            )
            merged.extend((item, base_url) for item in list(links or []))
        out: list[ExtractRef] = []
        seen: set[str] = set()
        resolved_relative_links = 0
        for item, base_url in merged:
            url, resolved_relative = self._normalize_explore_url_with_meta(
                item.url,
                base_url=base_url,
            )
            if not url:
                continue
            if resolved_relative:
                resolved_relative_links += 1
            key = canonicalize_url(url) or url.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                item.model_copy(
                    update={
                        "url": url,
                        "text": item.text.strip(),
                    },
                    deep=True,
                )
            )
        if ctx is not None and resolved_relative_links > 0:
            ctx.run.explore_resolved_relative_links += resolved_relative_links
        return out

    def _merge_and_dedupe_urls(self, urls: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in urls:
            url, _ = self._normalize_explore_url_with_meta(item)
            if not url:
                continue
            key = canonicalize_url(url) or url.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(url)
        return out

    def _filter_already_fetched_urls(
        self,
        *,
        ctx: RoundStepContext,
        urls: list[str],
    ) -> list[str]:
        self._synchronize_corpus_indexes(ctx=ctx)
        fetched_canonical_urls: set[str] = set()
        for item in ctx.knowledge.source_ids_by_url:
            token = clean_whitespace(item)
            if token:
                fetched_canonical_urls.add(token)
        out: list[str] = []
        seen: set[str] = set()
        for item in urls:
            url, _ = self._normalize_explore_url_with_meta(item)
            if not url:
                continue
            key = canonicalize_url(url) or url.casefold()
            if key in seen or key in fetched_canonical_urls:
                continue
            seen.add(key)
            out.append(url)
        return out

    def _normalize_explore_url(self, raw_url: str, *, base_url: str = "") -> str:
        return self._normalize_explore_url_with_meta(raw_url, base_url=base_url)[0]

    def _normalize_explore_url_with_meta(
        self, raw_url: str, *, base_url: str = ""
    ) -> tuple[str, bool]:
        token = clean_whitespace(raw_url)
        if not token or token.startswith("#"):
            return "", False
        resolved_relative = False
        if token.startswith("//"):
            token = f"https:{token}"
        elif base_url:
            try:
                parsed_raw = urlsplit(token)
            except Exception:  # noqa: S112
                parsed_raw = None
            if parsed_raw is not None and (
                not parsed_raw.scheme or not parsed_raw.netloc
            ):
                resolved_relative = True
            token = urljoin(base_url, token)
        try:
            parsed = urlsplit(token)
        except Exception:  # noqa: S112
            return "", False
        scheme = clean_whitespace(parsed.scheme).casefold()
        host = clean_whitespace(parsed.netloc)
        path = clean_whitespace(parsed.path or "/")
        if scheme not in _EXPLORE_ALLOWED_SCHEMES:
            return "", False
        if not host:
            return "", False
        if self._is_low_value_explore_path(path):
            return "", False
        return (
            urlunsplit(
                (
                    scheme,
                    host,
                    parsed.path or "/",
                    clean_whitespace(parsed.query),
                    "",
                )
            ),
            resolved_relative,
        )

    def _is_low_value_explore_path(self, path: str) -> bool:
        normalized = clean_whitespace(path).casefold().replace("_", "-")
        if not normalized or normalized == "/":
            return False
        segments = [seg for seg in normalized.split("/") if seg]
        for segment in segments:
            cleaned = segment.strip("-.")
            if not cleaned:
                continue
            if any(
                cleaned.startswith(prefix)
                for prefix in _EXPLORE_LOW_VALUE_PATH_PREFIXES
            ):
                return True
        return False

    def _build_explore_fetch_contexts(
        self,
        *,
        ctx: RoundStepContext,
        urls: list[str],
    ) -> list[FetchStepContext]:
        max_chars = ctx.run.limits.fetch_page_max_chars
        fetch_cfg = self.settings.fetch
        main_links_limit = max(1, int(fetch_cfg.extract.link_max_count))
        request = FetchRequest(
            urls=list(urls),
            crawl_mode="fallback",
            crawl_timeout=30.0,
            content=FetchContentRequest(
                detail="full",
                max_chars=max_chars,
                include_markdown_links=False,
                include_html_tags=False,
            ),
            abstracts=False,
            subpages=None,
            overview=True,
            others=FetchOthersRequest(max_links=main_links_limit),
        )
        contexts: list[FetchStepContext] = []
        round_index = (
            ctx.run.current.round_index
            if ctx.run.current is not None
            else ctx.run.round_index
        )
        for index, url in enumerate(urls):
            fetch_ctx = FetchStepContext(
                request=request.model_copy(update={"urls": [url]}),
                response=FetchResponse(
                    request_id=(
                        f"{ctx.request_id}:research:explore:{round_index}:{index}"
                    ),
                    results=[],
                    statuses=[],
                ),
                request_id=(f"{ctx.request_id}:research:explore:{round_index}:{index}"),
                url=url,
                url_index=index,
            )
            fetch_ctx.related.enabled = True
            fetch_ctx.page.crawl_mode = request.crawl_mode
            fetch_ctx.page.crawl_timeout_s = float(request.crawl_timeout or 0.0)
            fetch_ctx.related.link_limit = main_links_limit
            fetch_ctx.related.image_limit = None
            fetch_ctx.related.subpages.candidate_limit = (
                self._derive_subpage_links_limit(main_links_limit)
            )
            contexts.append(fetch_ctx)
        return contexts

    def _trim_candidate_to_fetch_budget(
        self,
        *,
        candidate: SearchFetchedCandidate,
        remaining_fetch_budget: int,
    ) -> tuple[SearchFetchedCandidate, int]:
        pages_left = max(0, remaining_fetch_budget)
        if pages_left <= 0:
            return (
                candidate.model_copy(
                    update={
                        "result": candidate.result.model_copy(update={"subpages": []}),
                        "subpage_links": [],
                    }
                ),
                0,
            )
        allowed_subpages = max(0, pages_left - 1)
        trimmed_subpages = list(candidate.result.subpages or [])[:allowed_subpages]
        trimmed_subpage_links = list(candidate.subpage_links or [])[:allowed_subpages]
        trimmed = candidate.model_copy(
            update={
                "result": candidate.result.model_copy(
                    update={"subpages": trimmed_subpages}
                ),
                "subpage_links": trimmed_subpage_links,
            }
        )
        consumed = 1 + len(trimmed_subpages)
        return trimmed, consumed

    def _select_candidates_for_fetch_budget(
        self,
        *,
        candidates: list[SearchFetchedCandidate],
        remaining_fetch_budget: int,
    ) -> tuple[list[SearchFetchedCandidate], list[SearchFetchedCandidate], int]:
        pages_left = max(0, remaining_fetch_budget)
        if pages_left <= 0 or not candidates:
            return [], [item.model_copy(deep=True) for item in candidates], 0
        main_selected = min(len(candidates), pages_left)
        selected_slots: list[int] = [
            1 if idx < main_selected else 0 for idx in range(len(candidates))
        ]
        pages_left -= main_selected
        subpage_indexes: list[int] = [0] * len(candidates)
        while pages_left > 0:
            progressed = False
            for idx, candidate in enumerate(candidates):
                available_subpages = list(candidate.result.subpages or [])
                if selected_slots[idx] <= 0 or subpage_indexes[idx] >= len(
                    available_subpages
                ):
                    continue
                selected_slots[idx] += 1
                subpage_indexes[idx] += 1
                pages_left -= 1
                progressed = True
                if pages_left <= 0:
                    break
            if not progressed:
                break
        selected: list[SearchFetchedCandidate] = []
        deferred: list[SearchFetchedCandidate] = []
        consumed = 0
        for idx, candidate in enumerate(candidates):
            pages_for_candidate = selected_slots[idx]
            if pages_for_candidate <= 0:
                deferred.append(candidate.model_copy(deep=True))
                continue
            trimmed_candidate, candidate_consumed = (
                self._trim_candidate_to_fetch_budget(
                    candidate=candidate,
                    remaining_fetch_budget=pages_for_candidate,
                )
            )
            consumed += candidate_consumed
            selected.append(trimmed_candidate)
            remaining_subpages = list(candidate.result.subpages or [])[
                max(0, pages_for_candidate - 1) :
            ]
            remaining_subpage_links = list(candidate.subpage_links or [])[
                max(0, pages_for_candidate - 1) :
            ]
            if remaining_subpages:
                deferred.append(
                    candidate.model_copy(
                        update={
                            "result": candidate.result.model_copy(
                                update={"subpages": remaining_subpages}
                            ),
                            "subpage_links": remaining_subpage_links,
                        },
                        deep=True,
                    )
                )
        return selected, deferred, consumed

    def _append_sources_from_fetch_result(
        self,
        *,
        ctx: RoundStepContext,
        result: FetchResultItem,
        round_index: int,
    ) -> tuple[int, list[int], list[int]]:
        new_canonical_ids: list[int] = []
        new_version_ids: list[int] = []
        upserted = self._append_source_version(
            ctx=ctx,
            url=result.url,
            title=result.title,
            overview=str(result.overview or ""),
            content=str(result.content or ""),
            round_index=round_index,
            is_subpage=False,
        )
        if upserted.is_new_canonical:
            new_canonical_ids.append(upserted.source_id)
        if upserted.is_new_version:
            new_version_ids.append(upserted.source_id)
        for sub in list(result.subpages or []):
            sub_upserted = self._append_source_from_subpage(
                ctx=ctx,
                sub=sub,
                round_index=round_index,
            )
            if sub_upserted.is_new_canonical:
                new_canonical_ids.append(sub_upserted.source_id)
            if sub_upserted.is_new_version:
                new_version_ids.append(sub_upserted.source_id)
        return upserted.source_id, new_canonical_ids, new_version_ids

    def _append_source_from_subpage(
        self,
        *,
        ctx: RoundStepContext,
        sub: FetchSubpagesResult,
        round_index: int,
    ) -> ResearchCorpusUpsertResult:
        return self._append_source_version(
            ctx=ctx,
            url=sub.url,
            title=sub.title,
            overview=str(sub.overview or ""),
            content=str(sub.content or ""),
            round_index=round_index,
            is_subpage=True,
        )

    def _build_link_candidate_from_search_candidate(
        self,
        *,
        candidate: SearchFetchedCandidate,
    ) -> SearchFetchedCandidate:
        return candidate.model_copy(deep=True)

    def _build_link_candidate_from_explore_result(
        self,
        *,
        result: FetchResultItem,
    ) -> SearchFetchedCandidate:
        raw_links = [
            ExtractRef(url=clean_whitespace(url))
            for url in list((result.others.links if result.others else []) or [])
        ]
        candidate = SearchFetchedCandidate(
            result=result.model_copy(deep=True),
            links=raw_links,
            subpage_links=[],
        )
        return candidate.model_copy(
            update={
                "links": self._merge_and_dedupe_links(candidate=candidate),
            },
            deep=True,
        )

    def _derive_subpage_links_limit(self, main_links_limit: int) -> int:
        return max(8, round(main_links_limit * 0.30))

    def _synchronize_corpus_indexes(self, *, ctx: RoundStepContext) -> None:
        """Synchronize corpus indexes in the knowledge state."""
        sorted_sources = sorted(ctx.knowledge.sources, key=lambda item: item.source_id)
        url_to_ids: dict[str, list[int]] = {}
        for idx, source in enumerate(sorted_sources):
            canonical = source.canonical_url or canonicalize_url(source.url)
            if canonical != source.canonical_url:
                source = source.model_copy(update={"canonical_url": canonical})
                sorted_sources[idx] = source
            if not canonical:
                continue
            ids = list(url_to_ids.get(canonical, []))
            ids.append(source.source_id)
            url_to_ids[canonical] = ids
        ctx.knowledge.sources = sorted_sources
        ctx.knowledge.source_ids_by_url = url_to_ids

    def _append_source_version(
        self,
        *,
        ctx: RoundStepContext,
        url: str,
        title: str,
        overview: str,
        content: str,
        round_index: int,
        is_subpage: bool,
        source_id: int | None = None,
    ) -> ResearchCorpusUpsertResult:
        """Append or update a source version in the knowledge corpus.

        Args:
            ctx: The round step context containing knowledge.
            url: The source URL.
            title: The source title.
            overview: The source overview/summary text.
            content: The full source content.
            round_index: The current round index.
            is_subpage: Whether this is a subpage.
            source_id: Pre-allocated unique source ID. If None, generates from list length
                (not safe for parallel tracks; prefer passing pre-allocated ID).

        Returns:
            ResearchCorpusUpsertResult with source_id and status flags.
        """
        normalized_url = clean_whitespace(url)
        canonical_url = canonicalize_url(normalized_url) or normalized_url
        normalized_title = clean_whitespace(title)
        normalized_overview = self._normalize_text(overview)
        normalized_content = self._normalize_text(content)
        fingerprint = self._build_content_fingerprint(
            content=normalized_content,
            overview=normalized_overview,
        )
        self._synchronize_corpus_indexes(ctx=ctx)
        source_idx_by_id = self._build_source_idx_by_id(ctx.knowledge.sources)
        existing_ids = list(ctx.knowledge.source_ids_by_url.get(canonical_url, []))
        for existing_source_id in reversed(existing_ids):
            idx = source_idx_by_id.get(existing_source_id)
            if idx is None:
                continue
            source = ctx.knowledge.sources[idx]
            if source.round_index != round_index:
                continue
            if source.content_fingerprint != fingerprint:
                continue
            updated = source.model_copy(
                update={
                    "title": source.title or normalized_title,
                    "overview": source.overview or normalized_overview,
                    "content": source.content or normalized_content,
                    "seen_count": max(1, source.seen_count) + 1,
                    "canonical_url": canonical_url,
                    "content_fingerprint": fingerprint,
                }
            )
            ctx.knowledge.sources[idx] = updated
            return ResearchCorpusUpsertResult(
                source_id=existing_source_id,
                canonical_url=canonical_url,
                is_new_canonical=False,
                is_new_version=False,
            )
        allocated_source_id = (
            source_id if source_id is not None else len(ctx.knowledge.sources) + 1
        )
        ctx.knowledge.sources.append(
            ResearchSource(
                source_id=allocated_source_id,
                url=normalized_url,
                canonical_url=canonical_url,
                title=normalized_title,
                overview=normalized_overview,
                content=normalized_content,
                round_index=round_index,
                is_subpage=is_subpage,
                seen_count=1,
                content_fingerprint=fingerprint,
            )
        )
        ids = list(ctx.knowledge.source_ids_by_url.get(canonical_url, []))
        ids.append(allocated_source_id)
        ctx.knowledge.source_ids_by_url[canonical_url] = ids
        return ResearchCorpusUpsertResult(
            source_id=allocated_source_id,
            canonical_url=canonical_url,
            is_new_canonical=len(existing_ids) == 0,
            is_new_version=True,
        )

    def _rebuild_corpus_ranking(
        self,
        *,
        ctx: RoundStepContext,
        round_index: int,
    ) -> float:
        """Rebuild corpus ranking scores."""
        self._synchronize_corpus_indexes(ctx=ctx)
        old_scores = dict(ctx.knowledge.source_scores)
        old_ranked = list(ctx.knowledge.ranked_source_ids)
        source_idx_by_id = self._build_source_idx_by_id(ctx.knowledge.sources)
        canonical_to_latest: dict[str, int] = {}
        canonical_seen: dict[str, int] = {}
        for source in ctx.knowledge.sources:
            canonical = source.canonical_url
            if not canonical:
                canonical = canonicalize_url(source.url)
            if not canonical:
                continue
            canonical_seen[canonical] = canonical_seen.get(canonical, 0) + max(
                1, source.seen_count
            )
            prev = canonical_to_latest.get(canonical)
            if prev is None:
                canonical_to_latest[canonical] = source.source_id
                continue
            prev_source = ctx.knowledge.sources[source_idx_by_id[prev]]
            if (
                source.round_index,
                source.source_id,
            ) >= (
                prev_source.round_index,
                prev_source.source_id,
            ):
                canonical_to_latest[canonical] = source.source_id
        max_seen = max(canonical_seen.values(), default=1)
        query_tokens = self._build_query_tokens(ctx=ctx)
        scored: list[tuple[int, float]] = []
        score_map: dict[int, float] = {}
        for canonical, latest_id in canonical_to_latest.items():
            idx = source_idx_by_id.get(latest_id)
            if idx is None:
                continue
            source = ctx.knowledge.sources[idx]
            newness_score = self._compute_newness_score(
                source_round_index=source.round_index,
                current_round_index=round_index,
            )
            relevance_score = self._compute_relevance_score(
                source=source,
                query_tokens=query_tokens,
            )
            depth_score = self._compute_depth_score(source)
            authority_score = source_authority_score(source)
            stability_score = min(
                1.0,
                float(canonical_seen.get(canonical, 0)) / float(max_seen or 1),
            )
            final_score = (
                float(_CORPUS_SCORE_WEIGHT_NEWNESS) * newness_score
                + float(_CORPUS_SCORE_WEIGHT_RELEVANCE) * relevance_score
                + float(_CORPUS_SCORE_WEIGHT_DEPTH) * depth_score
                + float(_CORPUS_SCORE_WEIGHT_STABILITY) * stability_score
                + float(_CORPUS_SCORE_WEIGHT_AUTHORITY) * authority_score
            )
            score_map[latest_id] = float(final_score)
            scored.append((latest_id, float(final_score)))
        scored.sort(
            key=lambda item: (
                float(item[1]),
                self._source_round_index(ctx=ctx, source_id=item[0]),
                item[0],
            ),
            reverse=True,
        )
        ranked_source_ids = [source_id for source_id, _ in scored]
        full_score_map: dict[int, float] = {}
        for canonical, ids in ctx.knowledge.source_ids_by_url.items():
            canonical_score = 0.0
            latest_source_id = canonical_to_latest.get(canonical)
            if latest_source_id is not None:
                canonical_score = float(score_map.get(latest_source_id, 0.0))
            for source_id in ids:
                full_score_map[source_id] = canonical_score
        for idx, source in enumerate(ctx.knowledge.sources):
            canonical = source.canonical_url or canonicalize_url(source.url)
            if canonical != source.canonical_url:
                ctx.knowledge.sources[idx] = source.model_copy(
                    update={
                        "canonical_url": canonical,
                    }
                )
        ctx.knowledge.ranked_source_ids = list(ranked_source_ids)
        ctx.knowledge.source_scores = dict(full_score_map)
        old_total = sum(
            float(old_scores.get(source_id, 0.0)) for source_id in old_ranked
        )
        new_total = sum(
            float(ctx.knowledge.source_scores.get(source_id, 0.0))
            for source_id in ranked_source_ids
        )
        return max(0.0, float(new_total - old_total))

    def _build_content_fingerprint(self, *, content: str, overview: str) -> str:
        """Build a SHA-256 fingerprint for content."""
        normalized_content = self._normalize_text(content)
        lines = [normalized_content[:5000]]
        if overview:
            lines.append(overview[:5000])
        payload = "\n".join(lines).strip() or "__empty__"
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _normalize_text(self, raw: str) -> str:
        return raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _build_source_idx_by_id(self, sources: list[ResearchSource]) -> dict[int, int]:
        return {item.source_id: idx for idx, item in enumerate(sources)}

    def _source_round_index(self, *, ctx: RoundStepContext, source_id: int) -> int:
        for source in ctx.knowledge.sources:
            if source.source_id == source_id:
                return source.round_index
        return 0

    def _compute_newness_score(
        self,
        *,
        source_round_index: int,
        current_round_index: int,
    ) -> float:
        round_gap = max(0, current_round_index - source_round_index)
        if round_gap <= 0:
            return 1.0
        return max(0.0, 1.0 - (float(round_gap) / 5.0))

    def _compute_relevance_score(
        self,
        *,
        source: ResearchSource,
        query_tokens: set[str],
    ) -> float:
        if not query_tokens:
            return 0.0
        parts = [
            source.title,
            source.url,
            (source.overview)[:1800],
            self._normalize_text(source.content)[:1800],
        ]
        haystack = " ".join(parts).casefold()
        if not haystack:
            return 0.0
        hits = sum(1 for token in query_tokens if token in haystack)
        return min(1.0, float(hits) / float(len(query_tokens)))

    def _compute_depth_score(self, source: ResearchSource) -> float:
        content_len = len(self._normalize_text(source.content))
        content_score = min(1.0, float(content_len) / 2400.0)
        overview_len = len(source.overview)
        overview_score = min(1.0, float(overview_len) / 1200.0)
        return min(1.0, 0.7 * content_score + 0.3 * overview_score)

    def _build_query_tokens(self, *, ctx: RoundStepContext) -> set[str]:
        tokens: set[str] = set()
        values: list[QuerySourceSpec | str] = list(ctx.run.next_queries)
        if ctx.run.current is not None:
            values.extend(ctx.run.current.queries)
        values.append(ctx.task.question)
        for value in values:
            query = value.query if isinstance(value, QuerySourceSpec) else value
            for token in _TOKEN_PATTERN.findall(query.casefold()):
                if len(token) < 2:
                    continue
                tokens.add(token)
        return tokens


def _score_at(*, scores: list[float], idx: int) -> float:
    return float(scores[idx]) if idx < len(scores) else 0.0


__all__ = ["ResearchFetchStep"]
