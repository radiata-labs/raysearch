from __future__ import annotations

from datetime import UTC, datetime
from typing_extensions import override

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
)
from raysearch.models.components.extract import ExtractRef
from raysearch.models.steps.fetch import FetchStepContext
from raysearch.models.steps.research import ResearchSource, RoundStepContext
from raysearch.models.steps.research.payloads import ResearchLinkPickerPayload
from raysearch.models.steps.search import SearchFetchedCandidate
from raysearch.steps.base import RunnerBase, StepBase
from raysearch.steps.research.prompt import build_link_picker_prompt_messages
from raysearch.steps.research.schema import build_link_picker_schema
from raysearch.steps.research.utils import resolve_research_model
from raysearch.tokenize import tokenize_for_query

_LINK_PICKER_PRERANK_MULTIPLIER = 4
_LINK_PICKER_PRERANK_FLOOR = 16


class ResearchFetchStep(StepBase[RoundStepContext]):
    fetch_runner: RunnerBase[FetchStepContext] = Depends(FETCH_RUNNER)
    ranker: RankerBase = Depends()
    llm: LLMClientBase = Depends()

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
        self._refresh_source_state(ctx=ctx)
        ctx.run.allocation.fetch_used += max(0, per_round_fetch_calls)

    def _append_sources_from_fetch_result(
        self,
        *,
        ctx: RoundStepContext,
        result: FetchResultItem,
        round_index: int,
    ) -> int:
        main_source_id = self._append_source(
            ctx=ctx,
            url=result.url,
            title=result.title,
            overview=str(result.overview or ""),
            content=str(result.content or ""),
            round_index=round_index,
            is_subpage=False,
        )
        for sub in list(result.subpages or []):
            self._append_source(
                ctx=ctx,
                url=sub.url,
                title=sub.title,
                overview=str(sub.overview or ""),
                content=str(sub.content or ""),
                round_index=round_index,
                is_subpage=True,
            )
        return main_source_id

    def _append_source(
        self,
        *,
        ctx: RoundStepContext,
        url: str,
        title: str,
        overview: str,
        content: str,
        round_index: int,
        is_subpage: bool,
    ) -> int:
        source_id = len(ctx.knowledge.sources) + 1
        ctx.knowledge.sources.append(
            ResearchSource(
                source_id=source_id,
                url=url,
                canonical_url=url,
                title=title,
                overview=overview,
                content=content,
                round_index=round_index,
                is_subpage=is_subpage,
            )
        )
        return source_id

    def _refresh_source_state(self, *, ctx: RoundStepContext) -> None:
        source_ids_by_url: dict[str, list[int]] = {}
        for source in ctx.knowledge.sources:
            ids = list(source_ids_by_url.get(source.canonical_url or source.url, []))
            ids.append(source.source_id)
            source_ids_by_url[source.canonical_url or source.url] = ids
        ctx.knowledge.source_ids_by_url = source_ids_by_url
        ctx.knowledge.ranked_source_ids = [
            source.source_id for source in reversed(ctx.knowledge.sources)
        ]
        ctx.knowledge.source_scores = {
            source.source_id: 1.0 for source in ctx.knowledge.sources
        }

    @override
    async def should_run(self, ctx: RoundStepContext) -> bool:
        """Execute only for explore round action (not search).

        Note: First round (round_index <= 1) is always search, never explore.
        """
        if ctx.run.stop or ctx.run.current is None:
            return False
        if ctx.run.current.round_index <= 1:
            return False
        round_action = (ctx.run.current.round_action or "search").casefold()
        return round_action == "explore"

    @override
    async def run_inner(self, ctx: RoundStepContext) -> RoundStepContext:
        # Pre-condition: should_run() already verified round_action == "explore"
        assert ctx.run.current is not None
        executed = await self._run_explore_fetch(ctx)
        if executed:
            ctx.run.current.search_fetched_candidates = []
            return ctx
        # Fallback: no valid explore targets, reset to search mode
        ctx.run.current.round_action = "search"
        ctx.run.current.explore_target_source_ids = []
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
            main_source_id = self._append_sources_from_fetch_result(
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
            return selected
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
        return merged

    async def _select_links_fast(
        self,
        *,
        ctx: RoundStepContext,
        candidate: SearchFetchedCandidate,
        max_links: int,
    ) -> list[str]:
        links = self._collect_links(candidate=candidate)
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
        links = self._collect_links(candidate=candidate)
        if not links:
            return []
        cap = max(
            _LINK_PICKER_PRERANK_FLOOR,
            max(1, max_links) * _LINK_PICKER_PRERANK_MULTIPLIER,
        )
        if len(links) > cap:
            ranked_indexes = await self._rank_link_indexes(ctx=ctx, links=links)
            links = [
                links[idx].model_copy()
                for idx in ranked_indexes[:cap]
                if idx < len(links)
            ]
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
        for link_id in selected_ids:
            if link_id <= 0 or link_id > len(links):
                continue
            url = links[link_id - 1].url
            if not url:
                continue
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

    def _collect_links(
        self,
        *,
        candidate: SearchFetchedCandidate,
    ) -> list[ExtractRef]:
        # ExtractRef has only primitive fields (text, url); shallow copy suffices.
        out = [item.model_copy() for item in list(candidate.links or []) if item.url]
        for links in list(candidate.subpage_links or []):
            out.extend(item.model_copy() for item in list(links or []) if item.url)
        return out

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

    def _build_link_candidate_from_explore_result(
        self,
        *,
        result: FetchResultItem,
    ) -> SearchFetchedCandidate:
        raw_links = [
            ExtractRef(url=url)
            for url in list((result.others.links if result.others else []) or [])
            if url
        ]
        return SearchFetchedCandidate(
            result=result.model_copy(deep=True),
            links=raw_links,
            subpage_links=[],
        )

    def _derive_subpage_links_limit(self, main_links_limit: int) -> int:
        return max(8, round(main_links_limit * 0.30))


def _score_at(*, scores: list[float], idx: int) -> float:
    return float(scores[idx]) if idx < len(scores) else 0.0


__all__ = ["ResearchFetchStep"]
