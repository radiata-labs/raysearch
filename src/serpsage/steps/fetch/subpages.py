from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.app.request import FetchOthersRequest
from serpsage.app.response import FetchSubpagesResult
from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchStepContext, FetchStepOthers
from serpsage.steps.base import RunnerBase, StepBase

if TYPE_CHECKING:
    from serpsage.app.response import FetchResultItem
    from serpsage.components.rank.base import RankerBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class FetchSubpageStep(StepBase[FetchStepContext]):
    span_name = "step.fetch_subpages"

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
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        if not ctx.enable_others_and_subpages:
            return ctx
        if not ctx.subpages.subpages_enabled or ctx.subpages.subpages_max <= 0:
            return ctx
        candidates = list(ctx.subpages.subpages_links or [])
        if not candidates:
            return ctx

        try:
            scores = await self._ranker.score_texts(
                texts=[f"[{item.anchor_text}]({item.url})" for item in candidates],
                query=ctx.subpages.subpages_query,
                query_tokens=ctx.subpages.subpages_keywords,
            )
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="fetch_subpage_failed",
                    message=str(exc),
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "subpages",
                        "fatal": False,
                        "crawl_mode": ctx.others.crawl_mode,
                    },
                )
            )
            return ctx

        ranked_indexes = sorted(
            range(len(candidates)),
            key=lambda idx: (-_score_at(scores=scores, idx=idx), idx),
        )
        selected_urls = [
            candidates[idx].url
            for idx in ranked_indexes[: max(1, int(ctx.subpages.subpages_max))]
        ]
        if not selected_urls:
            return ctx

        max_parallel = min(
            max(1, int(self.settings.fetch.concurrency.global_limit)),
            max(1, len(selected_urls)),
        )
        sem = anyio.Semaphore(max_parallel)
        out: list[FetchSubpagesResult | None] = [None] * len(selected_urls)
        out_md_for_abstract: list[str | None] = [None] * len(selected_urls)
        out_overview_scores: list[list[float] | None] = [None] * len(selected_urls)

        async def run_one(index: int, url: str) -> None:
            async with sem:
                child_request = ctx.request.model_copy(
                    update={
                        "urls": [url],
                        "subpages": None,
                        "others": FetchOthersRequest(),
                    }
                )
                child_context = await self._fetch_runner.run(
                    FetchStepContext(
                        settings=ctx.settings,
                        request=child_request,
                        request_id=ctx.request_id,
                        url=url,
                        url_index=index,
                        enable_others_and_subpages=False,
                        others=FetchStepOthers(
                            crawl_mode=child_request.crawl_mode,
                            crawl_timeout_s=float(child_request.crawl_timeout or 0.0),
                            max_links=None,
                            max_image_links=None,
                        ),
                    )
                )
                if child_context.result is None:
                    message = (
                        str(child_context.errors[0].message)
                        if child_context.errors
                        else "subpage fetch failed"
                    )
                    ctx.errors.append(
                        AppError(
                            code="fetch_subpage_failed",
                            message=message,
                            details={
                                "url": ctx.url,
                                "url_index": ctx.url_index,
                                "subpage_url": url,
                                "stage": "subpages",
                                "fatal": False,
                                "crawl_mode": ctx.others.crawl_mode,
                            },
                        )
                    )
                    return
                out[index] = _to_subpage_result(child_context.result)
                out_md_for_abstract[index] = (
                    str(child_context.extracted.md_for_abstract or "")
                    if child_context.extracted is not None
                    else ""
                )
                out_overview_scores[index] = [
                    float(item.score)
                    for item in list(child_context.overview_scored_abstracts or [])
                ]

        async with anyio.create_task_group() as tg:
            for index, url in enumerate(selected_urls):
                tg.start_soon(run_one, index, url)

        paired = [
            (item, md_text or "", overview_scores or [])
            for item, md_text, overview_scores in zip(
                out,
                out_md_for_abstract,
                out_overview_scores,
                strict=False,
            )
            if item is not None
        ]
        ctx.subpages_result = [item for item, _, _ in paired]
        ctx.subpages_md_for_abstract = [md_text for _, md_text, _ in paired]
        ctx.subpages_overview_scores = [
            overview_scores for _, _, overview_scores in paired
        ]
        span.set_attr("subpages_candidates_count", int(len(candidates)))
        span.set_attr("subpages_selected_count", int(len(selected_urls)))
        span.set_attr("subpages_success_count", int(len(ctx.subpages_result)))
        span.set_attr(
            "subpages_failure_count",
            int(len(selected_urls) - len(ctx.subpages_result)),
        )
        return ctx


def _to_subpage_result(value: FetchResultItem) -> FetchSubpagesResult:
    return FetchSubpagesResult(
        url=str(value.url),
        title=str(value.title),
        content=str(value.content),
        abstracts=[str(item) for item in list(value.abstracts or [])],
        abstract_scores=[float(item) for item in list(value.abstract_scores or [])],
        overview=value.overview,
    )


def _score_at(*, scores: list[float], idx: int) -> float:
    return float(scores[idx]) if idx < len(scores) else 0.0


__all__ = ["FetchSubpageStep"]
