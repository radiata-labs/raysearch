from __future__ import annotations

from serpsage.app.container import Container
from serpsage.app.response import Citation, OverviewResult
from serpsage.contracts.errors import AppError
from serpsage.overview.schema import overview_json_schema
from serpsage.pipeline.steps import StepContext


class OverviewStep:
    def __init__(self, container: Container) -> None:
        self._c = container

    async def run(self, ctx: StepContext) -> StepContext:
        span = self._c.telemetry.start_span("step.overview")
        try:
            enabled = ctx.settings.overview.enabled
            if ctx.request.overview is not None:
                enabled = bool(ctx.request.overview)
            if not enabled:
                return ctx
            if not ctx.results:
                return ctx
            if not ctx.settings.overview.llm.api_key:
                ctx.errors.append(
                    AppError(
                        code="overview_skipped",
                        message="LLM api_key not configured; skipping overview",
                        details={},
                    )
                )
                return ctx

            messages = _build_messages(ctx)
            schema = overview_json_schema()
            llm = ctx.settings.overview.llm
            try:
                data = await self._c.llm.chat_json(
                    model=llm.model,
                    messages=messages,
                    schema=schema,
                    timeout_s=float(llm.timeout_s),
                )
                ctx.overview = OverviewResult.model_validate(data)
            except Exception as exc:  # noqa: BLE001
                ctx.errors.append(
                    AppError(code="overview_failed", message=str(exc), details={})
                )
            return ctx
        finally:
            span.end()


def _build_messages(ctx: StepContext) -> list[dict[str, str]]:
    max_sources = int(ctx.settings.overview.max_sources)
    max_chunks = int(ctx.settings.overview.max_chunks_per_source)
    max_chunk_chars = int(ctx.settings.overview.max_chunk_chars)

    sources = []
    for r in ctx.results[:max_sources]:
        sid = r.source_id or "S?"
        parts = []
        if r.title:
            parts.append(f"TITLE: {r.title}")
        if r.url:
            parts.append(f"URL: {r.url}")
        if r.snippet:
            parts.append(f"SNIPPET: {r.snippet}")
        if r.page and r.page.chunks:
            for i, ch in enumerate(r.page.chunks[:max_chunks], 1):
                t = ch.text
                if max_chunk_chars and len(t) > max_chunk_chars:
                    t = t[:max_chunk_chars].rstrip() + "..."
                cid = ch.chunk_id or f"{sid}:C{i}"
                parts.append(f"CHUNK {cid}: {t}")
        sources.append(f"[{sid}]\n" + "\n".join(parts))

    user = "\n\n".join(
        [
            f"USER_QUERY:\n{ctx.request.query}",
            "SOURCES:\n" + "\n\n".join(sources),
            "TASK:\nWrite a concise overview. Provide key_points as short bullets. Add citations for key claims.",
            "CITATION_FORMAT:\nCitations must reference source_id (e.g. S1) and optionally chunk_id (e.g. S1:C1).",
        ]
    )
    return [
        {"role": "system", "content": "You are a research assistant. Output JSON only."},
        {"role": "user", "content": user},
    ]


__all__ = ["OverviewStep"]

