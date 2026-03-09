from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.components.base import Depends
from serpsage.components.fetch.base import FetchConfigBase
from serpsage.components.rank.base import RankerBase
from serpsage.core.runtime import Runtime
from serpsage.models.app.request import FetchOthersRequest
from serpsage.models.app.response import FetchResponse, FetchSubpagesResult
from serpsage.models.steps.fetch import FetchStepContext, FetchSubpageState
from serpsage.steps.base import RunnerBase, StepBase

if TYPE_CHECKING:
    from serpsage.models.app.response import FetchResultItem


class FetchSubpageStep(StepBase[FetchStepContext]):
    def __init__(
        self,
        *,
        rt: Runtime,
        fetch_runner: RunnerBase[FetchStepContext] = Depends(),
        ranker: RankerBase = Depends(),
    ) -> None:
        super().__init__(rt=rt)
        self._fetch_runner = fetch_runner
        self._ranker = ranker
        self.bind_deps(fetch_runner, ranker)

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        if not ctx.related.enabled:
            return ctx
        if not ctx.related.subpages.enabled or ctx.related.subpages.limit <= 0:
            return ctx
        candidates = list(ctx.related.subpages.candidates or [])
        if not candidates:
            return ctx
        try:
            scores = await self._ranker.score_texts(
                [f"[{item.text}]({item.url})" for item in candidates],
                query=ctx.related.subpages.query,
                query_tokens=ctx.related.subpages.keywords,
            )
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="fetch.subpages.error",
                request_id=ctx.request_id,
                stage="subpages",
                status="error",
                error_code="fetch_subpage_failed",
                error_type=type(exc).__name__,
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": False,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": str(exc),
                },
            )
            return ctx
        ranked_indexes = sorted(
            range(len(candidates)),
            key=lambda idx: (-_score_at(scores=scores, idx=idx), idx),
        )
        selected_urls = [
            candidates[idx].url
            for idx in ranked_indexes[: max(1, int(ctx.related.subpages.limit))]
        ]
        if not selected_urls:
            return ctx
        child_link_limit = int(ctx.related.subpages.candidate_limit or 0)
        if child_link_limit <= 0:
            child_link_limit = 8
        fetch_cfg = self.components.resolve_default_config(
            "fetch", expected_type=FetchConfigBase
        )
        child_link_limit = min(
            int(fetch_cfg.extract.link_max_count),
            max(1, int(child_link_limit)),
        )
        to_fetch: list[FetchStepContext] = []
        for index, url in enumerate(selected_urls):
            child_request = ctx.request.model_copy(
                update={
                    "urls": [url],
                    "subpages": None,
                    "others": FetchOthersRequest(max_links=child_link_limit),
                }
            )
            child_ctx = FetchStepContext(
                settings=ctx.settings,
                request=child_request,
                request_id=ctx.request_id,
                response=FetchResponse(
                    request_id=ctx.request_id,
                    results=[],
                    statuses=[],
                ),
                url=url,
                url_index=index,
            )
            child_ctx.related.enabled = True
            child_ctx.page.crawl_mode = child_request.crawl_mode
            child_ctx.page.crawl_timeout_s = float(child_request.crawl_timeout or 0.0)
            child_ctx.related.link_limit = int(child_link_limit)
            child_ctx.related.image_limit = None
            child_ctx.related.subpages.candidate_limit = None
            to_fetch.append(child_ctx)
        child_results = await self._fetch_runner.run_batch(to_fetch)
        out_items: list[FetchSubpageState | None] = [None] * len(selected_urls)
        for index, child_context in enumerate(child_results):
            url = selected_urls[index]
            if child_context.result is None or child_context.error.failed:
                await self.emit_tracking_event(
                    event_name="fetch.subpages.error",
                    request_id=ctx.request_id,
                    stage="subpages",
                    status="error",
                    error_code="fetch_subpage_failed",
                    attrs={
                        "url": ctx.url,
                        "url_index": int(ctx.url_index),
                        "subpage_url": url,
                        "fatal": False,
                        "crawl_mode": str(ctx.page.crawl_mode),
                        "message": "subpage fetch failed",
                    },
                )
                continue
            out_items[index] = FetchSubpageState(
                url=url,
                result=_to_subpage_result(child_context.result),
                doc=(
                    child_context.page.doc.model_copy(deep=True)
                    if child_context.page.doc is not None
                    else None
                ),
                overview_scores=[
                    float(item.score)
                    for item in list(child_context.analysis.overview.ranked or [])
                ],
            )
        ctx.related.subpages.items = [item for item in out_items if item is not None]
        return ctx


def _to_subpage_result(value: FetchResultItem) -> FetchSubpagesResult:
    return FetchSubpagesResult(
        url=str(value.url),
        title=str(value.title),
        published_date=str(value.published_date),
        author=str(value.author),
        image=str(value.image),
        favicon=str(value.favicon),
        content=str(value.content),
        abstracts=[str(item) for item in list(value.abstracts or [])],
        abstract_scores=[float(item) for item in list(value.abstract_scores or [])],
        overview=value.overview,
    )


def _score_at(*, scores: list[float], idx: int) -> float:
    return float(scores[idx]) if idx < len(scores) else 0.0


__all__ = ["FetchSubpageStep"]
