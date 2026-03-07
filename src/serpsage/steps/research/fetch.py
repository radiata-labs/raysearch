from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from typing_extensions import override
from urllib.parse import urljoin, urlsplit, urlunsplit

import anyio

from serpsage.models.app.request import (
    FetchContentRequest,
    FetchOthersRequest,
    FetchRequest,
)
from serpsage.models.app.response import FetchResultItem, FetchSubpagesResult
from serpsage.models.components.extract import ExtractedLink
from serpsage.models.steps.fetch import FetchRuntimeConfig, FetchStepContext
from serpsage.models.steps.research import (
    ResearchCorpusUpsertResult,
    ResearchLinkCandidate,
    ResearchLinkPickerPayload,
    ResearchStepContext,
)
from serpsage.models.steps.search import SearchFetchedCandidate
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_link_picker_prompt_messages
from serpsage.steps.research.schema import build_link_picker_schema
from serpsage.steps.research.search import (
    append_source_version,
    canonicalize_url,
    rebuild_corpus_ranking,
    synchronize_corpus_indexes,
)
from serpsage.steps.research.utils import resolve_research_model
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.components.rank.base import RankerBase
    from serpsage.core.runtime import Runtime
    from serpsage.steps.base import RunnerBase

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


class ResearchFetchStep(StepBase[ResearchStepContext]):
    def __init__(
        self,
        *,
        rt: Runtime,
        fetch_runner: RunnerBase[FetchStepContext],
        ranker: RankerBase,
        llm: LLMClientBase,
        phase: Literal["pre", "post"] = "post",
    ) -> None:
        super().__init__(rt=rt)
        self._fetch_runner = fetch_runner
        self._ranker = ranker
        self._llm = llm
        phase_key = phase.casefold()
        if phase_key not in {"pre", "post"}:
            phase_key = "post"
        self._phase: Literal["pre", "post"] = "pre" if phase_key == "pre" else "post"
        self.bind_deps(fetch_runner, ranker, llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        if ctx.runtime.stop or ctx.current_round is None:
            return ctx
        round_action = (ctx.work.round_action or "search").casefold()
        if ctx.current_round.round_index <= 1:
            round_action = "search"
        if self._phase == "pre":
            if round_action == "explore":
                executed = await self._run_explore_fetch(ctx)
                if executed:
                    ctx.work.search_fetched_candidates = []
                    return ctx
                ctx.work.round_action = "search"
                ctx.work.explore_target_source_ids = []
            return ctx
        if round_action != "search":
            ctx.work.search_fetched_candidates = []
            return ctx
        ctx = self._run_search_fetch(ctx)
        ctx.work.search_fetched_candidates = []
        return ctx

    def _run_search_fetch(self, ctx: ResearchStepContext) -> ResearchStepContext:
        assert ctx.current_round is not None
        candidates = [
            item.model_copy(deep=True)
            for item in list(ctx.work.search_fetched_candidates)
        ]
        remaining_fetch_budget = max(
            0,
            ctx.runtime.budget.max_fetch_calls - ctx.runtime.fetch_calls,
        )
        if remaining_fetch_budget <= 0:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_fetch_calls"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "max_fetch_calls"
            return ctx
        all_results: list[FetchResultItem] = []
        new_source_ids: list[int] = []
        per_round_fetch_calls = 0
        round_link_candidates: list[ResearchLinkCandidate] = []
        for candidate in candidates:
            if remaining_fetch_budget <= 0:
                break
            trimmed_candidate, consumed_pages = self._trim_candidate_to_fetch_budget(
                candidate=candidate,
                remaining_fetch_budget=remaining_fetch_budget,
            )
            if consumed_pages <= 0:
                continue
            remaining_fetch_budget = max(0, remaining_fetch_budget - consumed_pages)
            per_round_fetch_calls += consumed_pages
            all_results.append(trimmed_candidate.result)
            main_source_id, canonical_ids, _ = self._append_sources_from_fetch_result(
                ctx=ctx,
                result=trimmed_candidate.result,
                round_index=ctx.current_round.round_index,
            )
            for source_id in canonical_ids:
                if source_id not in new_source_ids:
                    new_source_ids.append(source_id)
            round_link_candidates.append(
                self._build_link_candidate_from_search_candidate(
                    source_id=main_source_id,
                    candidate=trimmed_candidate,
                    round_index=ctx.current_round.round_index,
                )
            )
        self._finalize_round_state(
            ctx=ctx,
            result_count=len(all_results),
            new_source_ids=new_source_ids,
            per_round_fetch_calls=per_round_fetch_calls,
            next_round_link_candidates=round_link_candidates,
        )
        return ctx

    async def _run_explore_fetch(self, ctx: ResearchStepContext) -> bool:
        assert ctx.current_round is not None
        remaining_fetch_budget = max(
            0,
            ctx.runtime.budget.max_fetch_calls - ctx.runtime.fetch_calls,
        )
        if remaining_fetch_budget <= 0:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_fetch_calls"
            return False
        expected_candidate_round = ctx.current_round.round_index - 1
        if ctx.plan.last_round_link_candidates_round != expected_candidate_round:
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
        fetched = await self._fetch_runner.run_batch(fetch_contexts)
        all_results: list[FetchResultItem] = []
        new_source_ids: list[int] = []
        per_round_fetch_calls = 0
        next_round_link_candidates: list[ResearchLinkCandidate] = []
        for item in fetched:
            if item.fatal or item.output.result is None:
                continue
            result = item.output.result
            per_round_fetch_calls += 1
            all_results.append(result)
            main_source_id, canonical_ids, _ = self._append_sources_from_fetch_result(
                ctx=ctx,
                result=result,
                round_index=ctx.current_round.round_index,
            )
            for source_id in canonical_ids:
                if source_id not in new_source_ids:
                    new_source_ids.append(source_id)
            next_round_link_candidates.append(
                self._build_link_candidate_from_explore_result(
                    source_id=main_source_id,
                    result=result,
                    round_index=ctx.current_round.round_index,
                )
            )
        if per_round_fetch_calls <= 0:
            return False
        if not next_round_link_candidates:
            next_round_link_candidates = [
                item.model_copy(
                    update={
                        "round_index": ctx.current_round.round_index,
                    },
                    deep=True,
                )
                for item in target_pages
            ]
        self._finalize_round_state(
            ctx=ctx,
            result_count=len(all_results),
            new_source_ids=new_source_ids,
            per_round_fetch_calls=per_round_fetch_calls,
            next_round_link_candidates=next_round_link_candidates,
        )
        return True

    def _finalize_round_state(
        self,
        *,
        ctx: ResearchStepContext,
        result_count: int,
        new_source_ids: list[int],
        per_round_fetch_calls: int,
        next_round_link_candidates: list[ResearchLinkCandidate],
    ) -> None:
        assert ctx.current_round is not None
        dedup_link_candidates = self._dedupe_link_candidates_by_source_id(
            next_round_link_candidates
        )
        ctx.plan.last_round_link_candidates = [
            item.model_copy(deep=True) for item in dedup_link_candidates
        ]
        ctx.plan.last_round_link_candidates_round = ctx.current_round.round_index
        ctx.current_round.result_count = max(0, result_count)
        ctx.current_round.new_source_ids = list(new_source_ids)
        ctx.current_round.corpus_score_gain = (
            float(
                rebuild_corpus_ranking(
                    ctx=ctx,
                    round_index=ctx.current_round.round_index,
                )
            )
            if ctx.current_round.result_count > 0
            else 0.0
        )
        ctx.runtime.fetch_calls += max(0, per_round_fetch_calls)

    def _select_explore_target_pages(
        self,
        *,
        ctx: ResearchStepContext,
    ) -> list[ResearchLinkCandidate]:
        candidates = list(ctx.plan.last_round_link_candidates or [])
        if not candidates:
            return []
        source_map = {item.source_id: item for item in candidates}
        target_ids = list(ctx.work.explore_target_source_ids or [])
        if not target_ids:
            return []
        cap = max(1, ctx.runtime.mode_depth.explore_target_pages_per_round)
        out: list[ResearchLinkCandidate] = []
        seen: set[int] = set()
        for source_id in target_ids:
            if source_id in seen:
                continue
            source = source_map.get(source_id)
            if source is None:
                continue
            seen.add(source_id)
            out.append(source.model_copy(deep=True))
            if len(out) >= cap:
                break
        return out

    async def _select_links_for_target_pages(
        self,
        *,
        ctx: ResearchStepContext,
        target_pages: list[ResearchLinkCandidate],
    ) -> list[str]:
        per_page_limit = max(1, ctx.runtime.mode_depth.explore_links_per_page)
        if ctx.runtime.mode_depth.mode_key == "research-fast":
            selected: list[str] = []
            for page in target_pages:
                selected.extend(
                    await self._select_links_fast(
                        ctx=ctx,
                        candidate=page,
                        max_links=per_page_limit,
                    )
                )
            return self._merge_and_dedupe_urls(selected)
        selected_by_page: list[list[str] | None] = [None] * len(target_pages)

        async def _pick_for_page(index: int, page: ResearchLinkCandidate) -> None:
            selected_by_page[index] = await self._select_links_with_llm(
                ctx=ctx,
                candidate=page,
                max_links=per_page_limit,
            )

        async with anyio.create_task_group() as tg:
            for idx, page in enumerate(target_pages):
                tg.start_soon(_pick_for_page, idx, page)
        merged: list[str] = []
        for links in selected_by_page:
            if not links:
                continue
            merged.extend(links)
        return self._merge_and_dedupe_urls(merged)

    async def _select_links_fast(
        self,
        *,
        ctx: ResearchStepContext,
        candidate: ResearchLinkCandidate,
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
        ctx: ResearchStepContext,
        candidate: ResearchLinkCandidate,
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
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        model = resolve_research_model(
            ctx=ctx,
            stage="link_select",
            fallback=fallback_model,
        )
        report_style = ctx.plan.theme_plan.report_style
        if report_style not in {"decision", "explainer", "execution"}:
            report_style = "explainer"
        candidate_links_markdown = self._render_link_candidates_for_picker(links)
        selected_ids: list[int] = []
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=build_link_picker_prompt_messages(
                    core_question=ctx.plan.theme_plan.core_question,
                    report_style=report_style,
                    mode_depth_profile=ctx.runtime.mode_depth.mode_key,
                    current_utc_date=datetime.fromtimestamp(
                        self.clock.now_ms() / 1000,
                        tz=UTC,
                    )
                    .date()
                    .isoformat(),
                    source_id=candidate.source_id,
                    source_url=candidate.url,
                    source_title=candidate.title,
                    max_links_to_select=max_links,
                    candidate_links_markdown=candidate_links_markdown,
                ),
                response_format=ResearchLinkPickerPayload,
                format_override=build_link_picker_schema(),
                retries=self.settings.research.llm_self_heal_retries,
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
        ctx: ResearchStepContext,
        links: list[ExtractedLink],
        max_links: int,
    ) -> list[ExtractedLink]:
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
        ctx: ResearchStepContext,
        links: list[ExtractedLink],
    ) -> list[int]:
        if not links:
            return []
        query = ctx.plan.theme_plan.core_question
        texts = [self._render_rank_text(item) for item in links]
        try:
            scores = await self._ranker.score_texts(
                texts=texts,
                query=query,
                query_tokens=tokenize_for_query(query),
            )
        except Exception:
            return list(range(len(links)))
        return sorted(
            range(len(links)),
            key=lambda idx: (-_score_at(scores=scores, idx=idx), idx),
        )

    def _render_rank_text(self, item: ExtractedLink) -> str:
        return f"anchor_text={item.anchor_text or '(none)'}; url={item.url}"

    def _render_link_candidates_for_picker(self, links: list[ExtractedLink]) -> str:
        if not links:
            return "- (none)"
        lines: list[str] = []
        for index, item in enumerate(links, start=1):
            lines.append(
                f"- id={index} | "
                f"anchor_text={item.anchor_text or '(none)'} | "
                f"url={item.url or 'n/a'}"
            )
        return "\n".join(lines).strip()

    def _merge_and_dedupe_links(
        self,
        *,
        ctx: ResearchStepContext | None = None,
        candidate: ResearchLinkCandidate,
    ) -> list[ExtractedLink]:
        merged: list[tuple[ExtractedLink, str]] = [
            (item, candidate.url) for item in list(candidate.links or [])
        ]
        for index, links in enumerate(list(candidate.subpage_links or [])):
            base_url = (
                candidate.subpage_urls[index]
                if index < len(candidate.subpage_urls)
                else candidate.url
            )
            merged.extend((item, base_url) for item in list(links or []))
        out: list[ExtractedLink] = []
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
                        "anchor_text": item.anchor_text.strip(),
                    },
                    deep=True,
                )
            )
        if ctx is not None and resolved_relative_links > 0:
            ctx.runtime.explore_resolved_relative_links += resolved_relative_links
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
        ctx: ResearchStepContext,
        urls: list[str],
    ) -> list[str]:
        synchronize_corpus_indexes(ctx=ctx)
        fetched_canonical_urls: set[str] = set()
        for item in ctx.corpus.source_url_to_ids:
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
        ctx: ResearchStepContext,
        urls: list[str],
    ) -> list[FetchStepContext]:
        max_chars = ctx.runtime.mode_depth.content_chars
        main_links_limit = max(1, self.settings.fetch.extract.link_max_count)
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
            ctx.current_round.round_index
            if ctx.current_round is not None
            else ctx.runtime.round_index
        )
        for index, url in enumerate(urls):
            contexts.append(
                FetchStepContext(
                    settings=ctx.settings,
                    request=request.model_copy(update={"urls": [url]}),
                    request_id=(
                        f"{ctx.request_id}:research:explore:{round_index}:{index}"
                    ),
                    url=url,
                    url_index=index,
                    runtime=FetchRuntimeConfig(
                        crawl_mode=request.crawl_mode,
                        crawl_timeout_s=float(request.crawl_timeout or 0.0),
                        max_links_for_subpages=self._derive_subpage_links_limit(
                            main_links_limit
                        ),
                        max_links=main_links_limit,
                        max_image_links=None,
                    ),
                )
            )
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

    def _append_sources_from_fetch_result(
        self,
        *,
        ctx: ResearchStepContext,
        result: FetchResultItem,
        round_index: int,
    ) -> tuple[int, list[int], list[int]]:
        new_canonical_ids: list[int] = []
        new_version_ids: list[int] = []
        upserted = append_source_version(
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
        ctx: ResearchStepContext,
        sub: FetchSubpagesResult,
        round_index: int,
    ) -> ResearchCorpusUpsertResult:
        return append_source_version(
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
        source_id: int,
        candidate: SearchFetchedCandidate,
        round_index: int,
    ) -> ResearchLinkCandidate:
        return ResearchLinkCandidate(
            source_id=source_id,
            url=candidate.result.url,
            title=candidate.result.title,
            links=[item.model_copy(deep=True) for item in list(candidate.links or [])],
            subpage_links=[
                [item.model_copy(deep=True) for item in list(group or [])]
                for group in list(candidate.subpage_links or [])
            ],
            subpage_urls=[
                clean_whitespace(item.url)
                for item in list(candidate.result.subpages or [])
            ],
            round_index=round_index,
        )

    def _build_link_candidate_from_explore_result(
        self,
        *,
        source_id: int,
        result: FetchResultItem,
        round_index: int,
    ) -> ResearchLinkCandidate:
        raw_links = [
            ExtractedLink(url=clean_whitespace(url))
            for url in list((result.others.links if result.others else []) or [])
        ]
        candidate = ResearchLinkCandidate(
            source_id=source_id,
            url=clean_whitespace(result.url),
            title=clean_whitespace(result.title),
            links=raw_links,
            subpage_links=[],
            subpage_urls=[],
            round_index=round_index,
        )
        return candidate.model_copy(
            update={
                "links": self._merge_and_dedupe_links(candidate=candidate),
                "subpage_links": [],
            },
            deep=True,
        )

    def _dedupe_link_candidates_by_source_id(
        self,
        candidates: list[ResearchLinkCandidate],
    ) -> list[ResearchLinkCandidate]:
        out: list[ResearchLinkCandidate] = []
        seen_source_ids: set[int] = set()
        for item in candidates:
            source_id = item.source_id
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            out.append(item.model_copy(deep=True))
        return out

    def _derive_subpage_links_limit(self, main_links_limit: int) -> int:
        return max(8, round(main_links_limit * 0.30))


def _score_at(*, scores: list[float], idx: int) -> float:
    return float(scores[idx]) if idx < len(scores) else 0.0


__all__ = ["ResearchFetchStep"]
