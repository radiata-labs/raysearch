from __future__ import annotations

import math
from typing import Literal, cast
from typing_extensions import override

from raysearch.models.steps.search import (
    QuerySourceSpec,
    SearchFetchState,
    SearchOutputState,
    SearchPlanState,
    SearchRankState,
    SearchRetrievalState,
    SearchStepContext,
)
from raysearch.settings.models import SearchModeSettings
from raysearch.steps.base import StepBase
from raysearch.utils import clean_whitespace


class SearchPrepareStep(StepBase[SearchStepContext]):
    @override
    async def should_run(self, ctx: SearchStepContext) -> bool:
        """Prepare always runs (initializes search context)."""
        _ = ctx
        return True

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        query = clean_whitespace(ctx.request.query or "")
        mode = self._normalize_mode(ctx.request.mode)
        max_results = (
            ctx.request.max_results
            if ctx.request.max_results is not None
            else self.settings.search.max_results
        )
        max_results = max(1, max_results)
        profile = self._resolve_mode_settings(mode=mode)
        prefetch_limit = self._resolve_prefetch_limit(
            max_results=max_results,
            prefetch_multiplier=profile.prefetch_multiplier,
            prefetch_max_urls=profile.prefetch_max_urls,
        )
        ctx.request = ctx.request.model_copy(
            update={
                "query": query,
                "mode": mode,
                "max_results": max_results,
            }
        )
        ctx.plan = SearchPlanState(
            mode=mode,
            max_results=max_results,
            max_extra_queries=profile.max_extra_queries,
            prefetch_limit=prefetch_limit,
            context_docs_limit=profile.context_docs_limit,
            context_doc_min_chars=profile.context_doc_min_chars,
            rank_by_context=profile.rank_by_context,
        )
        if not ctx.runtime.engine_selection_subsystem:
            ctx.runtime.engine_selection_subsystem = "search"
        if not ctx.runtime.additional_queries and ctx.request.additional_queries:
            ctx.runtime.additional_queries = [
                QuerySourceSpec(query=item) for item in ctx.request.additional_queries
            ]
        ctx.runtime.provider_limit = ctx.runtime.provider_limit or max(
            max_results,
            prefetch_limit,
        )
        ctx.retrieval = SearchRetrievalState()
        ctx.fetch = SearchFetchState()
        ctx.rank = SearchRankState()
        ctx.output = SearchOutputState()
        await self.tracker.info(
            name="search.prepare.completed",
            request_id=ctx.request_id,
            step="search.prepare",
            data={
                "mode": mode,
                "max_results": max_results,
                "prefetch_limit": prefetch_limit,
            },
        )
        return ctx

    def _resolve_mode_settings(self, *, mode: str) -> SearchModeSettings:
        profiles = self.settings.search.modes
        if mode == "fast":
            return profiles.fast
        if mode == "deep":
            return profiles.deep
        return profiles.auto

    def _resolve_prefetch_limit(
        self,
        *,
        max_results: int,
        prefetch_multiplier: float,
        prefetch_max_urls: int,
    ) -> int:
        desired = math.ceil(max_results * prefetch_multiplier)
        limited = min(prefetch_max_urls, desired)
        return max(1, max_results, limited)

    def _normalize_mode(self, value: object) -> Literal["fast", "auto", "deep"]:
        token = clean_whitespace(str(value or "")).casefold()
        if token in {"fast", "auto", "deep"}:
            return cast("Literal['fast', 'auto', 'deep']", token)
        return "auto"


__all__ = ["SearchPrepareStep"]
