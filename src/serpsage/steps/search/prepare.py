from __future__ import annotationsfrom typing import TYPE_CHECKINGfrom typing_extensions import overridefrom serpsage.models.pipeline import SearchDeepState, SearchRankState, SearchStepContextfrom serpsage.steps.base import StepBasefrom serpsage.utils import clean_whitespaceif TYPE_CHECKING:
    from serpsage.core.runtime import Runtime

class SearchPrepareStep(StepBase[SearchStepContext]):

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: SearchStepContext
    ) -> SearchStepContext:
        query = clean_whitespace(ctx.request.query or "")
        mode = ctx.request.mode or "auto"
        max_results = (
            int(ctx.request.max_results)
            if ctx.request.max_results is not None
            else int(self.settings.search.max_results)
        )
        max_results = max(1, int(max_results))
        ctx.request = ctx.request.model_copy(
            update={
                "query": query,
                "mode": mode,
                "max_results": max_results,
            }
        )
        ctx.deep = SearchDeepState()
        ctx.prefetch.urls = []
        ctx.prefetch.scores = {}
        ctx.fetch.candidates = []
        ctx.rank = SearchRankState()
        ctx.output.results = []
        return ctx

__all__ = ["SearchPrepareStep"]
