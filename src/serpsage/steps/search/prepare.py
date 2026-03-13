from __future__ import annotations

import math
from typing import Literal, cast
from typing_extensions import override

from serpsage.models.steps.search import (
    SearchFetchState,
    SearchOutputState,
    SearchPlanState,
    SearchRankState,
    SearchRetrievalState,
    SearchStepContext,
)
from serpsage.settings.models import SearchModeSettings
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace


class SearchPrepareStep(StepBase[SearchStepContext]):
    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        query = clean_whitespace(ctx.request.query or "")
        mode = self._normalize_mode(ctx.request.mode)
        max_results = (
            int(ctx.request.max_results)
            if ctx.request.max_results is not None
            else int(self.settings.search.max_results)
        )
        max_results = max(1, int(max_results))
        profile = self._resolve_mode_settings(mode=mode)
        prefetch_limit = self._resolve_prefetch_limit(
            max_results=max_results,
            prefetch_multiplier=float(profile.prefetch_multiplier),
            prefetch_max_urls=int(profile.prefetch_max_urls),
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
            max_extra_queries=int(profile.max_extra_queries),
            prefetch_limit=int(prefetch_limit),
            context_docs_limit=int(profile.context_docs_limit),
            context_doc_min_chars=int(profile.context_doc_min_chars),
            rank_by_context=bool(profile.rank_by_context),
            optimize_query=mode != "fast",
            optimized_query=query,
        )
        ctx.runtime.provider_limit = int(
            ctx.runtime.provider_limit or max(max_results, prefetch_limit)
        )
        ctx.retrieval = SearchRetrievalState()
        ctx.fetch = SearchFetchState()
        ctx.rank = SearchRankState()
        ctx.output = SearchOutputState()
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
        desired = int(math.ceil(float(max_results) * float(prefetch_multiplier)))
        limited = min(int(prefetch_max_urls), desired)
        return max(1, int(max_results), limited)

    def _normalize_mode(self, value: object) -> Literal["fast", "auto", "deep"]:
        token = clean_whitespace(str(value or "")).casefold()
        if token in {"fast", "auto", "deep"}:
            return cast("Literal['fast', 'auto', 'deep']", token)
        return "auto"


__all__ = ["SearchPrepareStep"]
