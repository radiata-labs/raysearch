from __future__ import annotations

import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from serpsage.core.runtime import Overrides
from serpsage.core.workunit import WorkUnit
from serpsage.models.app.response import (
    AnswerResponse,
    FetchResponse,
    FetchStatusError,
    FetchStatusItem,
    ResearchResponse,
    SearchResponse,
)
from serpsage.models.components.telemetry import MeterPayload
from serpsage.models.steps.answer import AnswerStepContext
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.models.steps.research import ResearchStepContext
from serpsage.models.steps.search import SearchStepContext
from serpsage.steps.base import RunnerBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.models.app.request import (
        AnswerRequest,
        FetchRequest,
        ResearchRequest,
        SearchRequest,
    )
    from serpsage.settings.models import AppSettings


class Engine(WorkUnit):
    """Async-only engine with search/fetch/answer paths."""

    def __init__(
        self,
        *,
        rt: Runtime,
        search_runner: RunnerBase[SearchStepContext],
        fetch_runner: RunnerBase[FetchStepContext],
        answer_runner: RunnerBase[AnswerStepContext],
        research_runner: RunnerBase[ResearchStepContext] | None = None,
    ) -> None:
        super().__init__(rt=rt)
        self._search_runner = search_runner
        self._fetch_runner = fetch_runner
        self._answer_runner = answer_runner
        self._research_runner = research_runner or RunnerBase[ResearchStepContext](
            rt=rt,
            steps=[],
            kind="search",
        )
        self.bind_deps(
            search_runner,
            fetch_runner,
            answer_runner,
            self._research_runner,
            self.telemetry,
        )

    @classmethod
    def from_settings(
        cls, settings: AppSettings, *, overrides: Overrides | None = None
    ) -> Engine:
        from serpsage.app.bootstrap import build_engine  # noqa: PLC0415

        return build_engine(settings=settings, overrides=overrides)

    async def search(self, req: SearchRequest) -> SearchResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self._emit_safe(
            event_name="request.start",
            status="start",
            request_id=request_id,
            component="engine",
            stage="search",
            attrs={"request_kind": "search"},
        )
        ctx = SearchStepContext(
            settings=self.settings,
            request=req,
            request_id=request_id,
            response=SearchResponse(
                request_id=request_id,
                search_mode=req.mode,
                results=[],
            ),
        )
        try:
            ctx = await self._search_runner.run(ctx)
            ctx.response.search_mode = ctx.request.mode
            ctx.response.results = list(ctx.output.results)
            response = ctx.response
            await self._emit_safe(
                event_name="request.end",
                status="ok",
                request_id=request_id,
                component="engine",
                stage="search",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "request_kind": "search",
                    "result_count": len(response.results),
                },
            )
            await self._emit_request_meter(request_id=request_id, stage="search")
            return response
        except Exception as exc:  # noqa: BLE001
            await self._emit_safe(
                event_name="request.error",
                status="error",
                request_id=request_id,
                component="engine",
                stage="search",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs={"request_kind": "search"},
            )
            await self._emit_request_meter(
                request_id=request_id,
                stage="search",
                status="error",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._pop_request_context(token)

    async def fetch(self, req: FetchRequest) -> FetchResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self._emit_safe(
            event_name="request.start",
            status="start",
            request_id=request_id,
            component="engine",
            stage="fetch",
            attrs={
                "request_kind": "fetch",
                "url_count": len(req.urls),
            },
        )
        try:
            contexts: list[FetchStepContext] = []
            for idx, url in enumerate(req.urls):
                fetch_ctx = FetchStepContext(
                    settings=self.settings,
                    request=req,
                    request_id=request_id,
                    response=FetchResponse(
                        request_id=request_id,
                        results=[],
                        statuses=[],
                    ),
                    url=url,
                    url_index=idx,
                )
                fetch_ctx.related.enabled = True
                fetch_ctx.page.crawl_mode = req.crawl_mode
                fetch_ctx.page.crawl_timeout_s = float(req.crawl_timeout or 0.0)
                fetch_ctx.related.link_limit = (
                    req.others.max_links if req.others is not None else None
                )
                fetch_ctx.related.image_limit = (
                    req.others.max_image_links if req.others is not None else None
                )
                contexts.append(fetch_ctx)
            if contexts:
                contexts = await self._fetch_runner.run_batch(contexts)
            results = [
                ctx.result
                for ctx in contexts
                if not ctx.error.failed and ctx.result is not None
            ]
            statuses = [
                FetchStatusItem(
                    url=str(url),
                    status="error",
                    error=FetchStatusError(
                        tag="SOURCE_NOT_AVAILABLE",
                        detail="not processed",
                    ),
                )
                for url in req.urls
            ]
            for ctx_item in contexts:
                idx = int(ctx_item.url_index)
                if idx < 0 or idx >= len(statuses):
                    continue
                success = bool(
                    ctx_item.result is not None and not ctx_item.error.failed
                )
                statuses[idx] = FetchStatusItem(
                    url=str(ctx_item.url),
                    status="success" if success else "error",
                    error=(
                        None
                        if success
                        else FetchStatusError(
                            tag=ctx_item.error.tag,
                            detail=ctx_item.error.detail,
                        )
                    ),
                )
            success_count = sum(1 for item in statuses if str(item.status) == "success")
            error_count = max(0, len(statuses) - success_count)
            response = FetchResponse(
                request_id=request_id,
                results=results,
                statuses=statuses,
            )
            await self._emit_safe(
                event_name="request.end",
                status="ok",
                request_id=request_id,
                component="engine",
                stage="fetch",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "request_kind": "fetch",
                    "result_count": len(response.results),
                    "success_count": int(success_count),
                    "error_count": int(error_count),
                    "url_count": len(req.urls),
                },
            )
            await self._emit_request_meter(request_id=request_id, stage="fetch")
            return response
        except Exception as exc:  # noqa: BLE001
            await self._emit_safe(
                event_name="request.error",
                status="error",
                request_id=request_id,
                component="engine",
                stage="fetch",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs={"request_kind": "fetch"},
            )
            await self._emit_request_meter(
                request_id=request_id,
                stage="fetch",
                status="error",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._pop_request_context(token)

    async def answer(self, req: AnswerRequest) -> AnswerResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self._emit_safe(
            event_name="request.start",
            status="start",
            request_id=request_id,
            component="engine",
            stage="answer",
            attrs={"request_kind": "answer"},
        )
        ctx = AnswerStepContext(
            settings=self.settings,
            request=req,
            request_id=request_id,
            response=AnswerResponse(
                request_id=request_id,
                answer={},
                citations=[],
            ),
        )
        try:
            ctx = await self._answer_runner.run(ctx)
            answer: str | object
            if isinstance(req.json_schema, dict):
                answer = (
                    ctx.output.answers if isinstance(ctx.output.answers, dict) else {}
                )
            else:
                answer = (
                    str(ctx.output.answers)
                    if isinstance(ctx.output.answers, str)
                    else ""
                )
            ctx.response.answer = answer
            ctx.response.citations = list(ctx.output.citations)
            response = ctx.response
            await self._emit_safe(
                event_name="request.end",
                status="ok",
                request_id=request_id,
                component="engine",
                stage="answer",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "request_kind": "answer",
                    "citation_count": len(response.citations),
                    "has_answer": bool(
                        response.answer if isinstance(response.answer, str) else True
                    ),
                },
            )
            await self._emit_request_meter(request_id=request_id, stage="answer")
            return response
        except Exception as exc:  # noqa: BLE001
            await self._emit_safe(
                event_name="request.error",
                status="error",
                request_id=request_id,
                component="engine",
                stage="answer",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs={"request_kind": "answer"},
            )
            await self._emit_request_meter(
                request_id=request_id,
                stage="answer",
                status="error",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._pop_request_context(token)

    async def research(self, req: ResearchRequest) -> ResearchResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self._emit_safe(
            event_name="request.start",
            status="start",
            request_id=request_id,
            component="engine",
            stage="research",
            attrs={"request_kind": "research"},
        )
        ctx = ResearchStepContext(
            settings=self.settings,
            request=req,
            request_id=request_id,
            response=ResearchResponse(
                request_id=request_id,
                content="",
                structured=None,
            ),
        )
        try:
            ctx = await self._research_runner.run(ctx)
            ctx.response.content = ctx.output.content
            ctx.response.structured = ctx.output.structured
            response = ctx.response
            await self._emit_safe(
                event_name="request.end",
                status="ok",
                request_id=request_id,
                component="engine",
                stage="research",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "request_kind": "research",
                    "content_chars": len(str(response.content or "")),
                    "has_structured": response.structured is not None,
                },
            )
            await self._emit_request_meter(request_id=request_id, stage="research")
            return response
        except Exception as exc:  # noqa: BLE001
            await self._emit_safe(
                event_name="request.error",
                status="error",
                request_id=request_id,
                component="engine",
                stage="research",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs={"request_kind": "research"},
            )
            await self._emit_request_meter(
                request_id=request_id,
                stage="research",
                status="error",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._pop_request_context(token)

    async def _emit_request_meter(
        self,
        *,
        request_id: str,
        stage: str,
        status: str = "ok",
        error_type: str = "",
    ) -> None:
        await self._emit_safe(
            event_name="meter.usage.request",
            status="error" if status == "error" else "ok",
            request_id=request_id,
            component="engine",
            stage=stage,
            error_type=error_type,
            idempotency_key=f"{request_id}:meter.usage.request:{stage}",
            attrs={"request_kind": stage},
            meter=MeterPayload(
                meter_type="request",
                unit="request",
                quantity=1.0,
            ),
        )

    async def _emit_safe(self, **kwargs: Any) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        with suppress(Exception):
            await telemetry.emit(**kwargs)

    def _push_request_context(self, request_id: str) -> object:
        telemetry = self.telemetry
        if telemetry is None:
            return None
        try:
            return telemetry.push_request_context(request_id=request_id)
        except Exception:
            return None

    def _pop_request_context(self, token: object) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        with suppress(Exception):
            telemetry.pop_request_context(token)


__all__ = ["Engine"]
