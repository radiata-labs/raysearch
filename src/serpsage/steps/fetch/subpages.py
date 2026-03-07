from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import FetchOthersRequest
from serpsage.app.response import FetchSubpagesResult
from serpsage.components.extract.models import ExtractedLink
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.steps.models import FetchRuntimeConfig, FetchStepContext

if TYPE_CHECKING:
    from serpsage.app.response import FetchResultItem
    from serpsage.components.rank.base import RankerBase
    from serpsage.core.runtime import Runtime


class FetchSubpageStep(StepBase[FetchStepContext]):
    def __init__(
        self,
        *,
        rt: Runtime,
        fetch_runner: RunnerBase[FetchStepContext],
        ranker: RankerBase,
    ) -> None:
        super().__init__(rt=rt)
        self._fetch_runner = fetch_runner
        self._ranker = ranker
        self.bind_deps(fetch_runner, ranker)

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        if not ctx.enable_others_and_subpages:
            return ctx
        if not ctx.subpages.enabled or ctx.subpages.max_count <= 0:
            return ctx
        candidates = list(ctx.subpages.links or [])
        if not candidates:
            return ctx
        try:
            scores = await self._ranker.score_texts(
                texts=[f"[{item.anchor_text}]({item.url})" for item in candidates],
                query=ctx.subpages.query,
                query_tokens=ctx.subpages.keywords,
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
                    "crawl_mode": str(ctx.runtime.crawl_mode),
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
            for idx in ranked_indexes[: max(1, int(ctx.subpages.max_count))]
        ]
        if not selected_urls:
            return ctx
        child_link_limit = int(ctx.runtime.max_links_for_subpages or 0)
        if child_link_limit <= 0:
            child_link_limit = 8
        child_link_limit = min(
            int(self.settings.fetch.extract.link_max_count),
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
            to_fetch.append(
                FetchStepContext(
                    settings=ctx.settings,
                    request=child_request,
                    request_id=ctx.request_id,
                    url=url,
                    url_index=index,
                    enable_others_and_subpages=True,
                    runtime=FetchRuntimeConfig(
                        crawl_mode=child_request.crawl_mode,
                        crawl_timeout_s=float(child_request.crawl_timeout or 0.0),
                        max_links_for_subpages=None,
                        max_links=int(child_link_limit),
                        max_image_links=None,
                    ),
                )
            )
        child_results = await self._fetch_runner.run_batch(to_fetch)
        out: list[FetchSubpagesResult | None] = [None] * len(selected_urls)
        out_md_for_abstract: list[str | None] = [None] * len(selected_urls)
        out_overview_scores: list[list[float] | None] = [None] * len(selected_urls)
        out_result_links: list[list[ExtractedLink] | None] = [None] * len(selected_urls)
        for index, child_context in enumerate(child_results):
            url = selected_urls[index]
            if child_context.output.result is None:
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
                        "crawl_mode": str(ctx.runtime.crawl_mode),
                        "message": "subpage fetch failed",
                    },
                )
                continue
            out[index] = _to_subpage_result(child_context.output.result)
            out_md_for_abstract[index] = (
                str(child_context.artifacts.extracted.md_for_abstract or "")
                if child_context.artifacts.extracted is not None
                else ""
            )
            out_overview_scores[index] = [
                float(item.score)
                for item in list(
                    child_context.artifacts.overview_scored_abstracts or []
                )
            ]
            out_result_links[index] = list(
                child_context.artifacts.extracted.links
                if child_context.artifacts.extracted is not None
                else []
            )
        paired = [
            (item, md_text or "", overview_scores or [], result_links or [])
            for item, md_text, overview_scores, result_links in zip(
                out,
                out_md_for_abstract,
                out_overview_scores,
                out_result_links,
                strict=False,
            )
            if item is not None
        ]
        ctx.subpages.results = [item for item, _, _, _ in paired]
        ctx.subpages.md_for_abstract = [md_text for _, md_text, _, _ in paired]
        ctx.subpages.overview_scores = [
            overview_scores for _, _, overview_scores, _ in paired
        ]
        ctx.subpages.result_links = [links for _, _, _, links in paired]
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
