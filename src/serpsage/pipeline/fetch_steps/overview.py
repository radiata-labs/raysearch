from __future__ import annotations

import hashlib
import re
import time
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from pydantic import ValidationError

from serpsage.app.response import (
    OverviewLLMOutput,
    OverviewResult,
    PageChunk,
    ResultItem,
)
from serpsage.components.overview.schema import overview_json_schema
from serpsage.models.errors import AppError
from serpsage.models.llm import LLMUsage
from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import CacheBase, LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.models.llm import ChatJSONResult

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")


class FetchOverviewStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_overview"

    def __init__(
        self,
        *,
        rt: Runtime,
        llm: LLMClientBase,
        cache: CacheBase,
    ) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self._cache = cache
        self.bind_deps(llm, cache)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        profile = self.settings.fetch.overview
        enabled = profile.enabled_default
        if ctx.request.overview is not None:
            enabled = bool(ctx.request.overview)
        if not enabled:
            return ctx

        source = self._build_source(ctx)
        if source is None:
            return ctx
        query = (ctx.request.query or "").strip() or ctx.request.url
        llm_cfg = self.settings.llm.resolve_model(profile.use_model)

        messages = self._build_messages(
            query=query,
            results=[source],
            max_sources=1,
            max_chunks_per_source=max(1, int(profile.max_chunks)),
            max_chunk_chars=int(profile.max_chunk_chars),
            max_prompt_chars=int(profile.max_prompt_chars),
            force_language=str(profile.force_language),
        )
        schema = self._schema()
        cache_ttl_s = int(profile.cache_ttl_s)
        cache_key: str | None = None
        if cache_ttl_s > 0:
            cache_key = hashlib.sha256(
                stable_json(
                    {
                        "use_model": str(profile.use_model),
                        "messages": messages,
                        "schema": schema,
                        "schema_strict": bool(llm_cfg.schema_strict),
                    }
                ).encode("utf-8")
            ).hexdigest()
            cached = await self._cache.aget(namespace="overview:fetch", key=cache_key)
            if cached:
                span.set_attr("cache_hit", True)
                try:
                    ctx.overview = OverviewResult.model_validate_json(cached)
                except Exception:
                    span.add_event("overview.cache_corrupt")
                else:
                    return ctx
        span.set_attr("cache_hit", False)

        try:
            t0 = time.monotonic()
            overview = await self._build_overview(
                query=query,
                results=[source],
                use_model=str(profile.use_model),
                max_sources=1,
                max_chunks_per_source=max(1, int(profile.max_chunks)),
                max_chunk_chars=int(profile.max_chunk_chars),
                max_prompt_chars=int(profile.max_prompt_chars),
                max_output_tokens=int(profile.max_output_tokens),
                self_heal_retries=int(profile.self_heal_retries),
                force_language=str(profile.force_language),
            )
            ctx.page.timing_ms["overview_ms"] = int((time.monotonic() - t0) * 1000)
            ctx.overview = overview
            if cache_ttl_s > 0 and cache_key:
                await self._cache.aset(
                    namespace="overview:fetch",
                    key=cache_key,
                    value=overview.model_dump_json().encode("utf-8"),
                    ttl_s=cache_ttl_s,
                )
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="fetch_overview_failed",
                    message=str(exc),
                    details={"type": type(exc).__name__},
                )
            )
        return ctx

    def _schema(self) -> dict[str, Any]:
        return overview_json_schema()

    def _build_messages(
        self,
        *,
        query: str,
        results: list[ResultItem],
        max_sources: int,
        max_chunks_per_source: int,
        max_chunk_chars: int,
        max_prompt_chars: int,
        force_language: str,
    ) -> list[dict[str, str]]:
        lang = force_language
        if lang == "auto":
            lang = "zh" if _CJK_RE.search(query or "") else "en"

        sources: list[str] = []
        for r in results[: max(1, int(max_sources))]:
            sid = r.source_id or "S?"
            parts: list[str] = []
            if r.title:
                parts.append(f"TITLE: {r.title}")
            if r.url:
                parts.append(f"URL: {r.url}")
            if r.snippet:
                parts.append(f"SNIPPET: {r.snippet}")
            if r.page and r.page.chunks:
                for i, ch in enumerate(
                    r.page.chunks[: max(1, int(max_chunks_per_source))], 1
                ):
                    t = ch.text
                    if max_chunk_chars and len(t) > max_chunk_chars:
                        t = t[:max_chunk_chars].rstrip() + "..."
                    cid = ch.chunk_id or f"{sid}:C{i}"
                    parts.append(f"CHUNK {cid}: {t}")
            sources.append(f"[{sid}]\n" + "\n".join(parts))

        if lang == "zh":
            task = (
                "TASK:\n"
                "Write a concise overview in Simplified Chinese (JSON). "
                "summary is a short paragraph; key_points is a list of short bullets; "
                "add citations for key claims."
            )
        else:
            task = (
                "TASK:\n"
                "Write a concise overview in English (JSON). "
                "summary is a short paragraph; key_points is a list of short bullets; "
                "add citations for key claims."
            )

        cite_fmt = (
            "CITATION_RULES:\n"
            "- citations[].cite_id must be unique, e.g. C1/C2...\n"
            "- citations[].source_id must reference provided SOURCES (e.g. S1)\n"
            "- citations[].chunk_id is optional, but if present must exist (e.g. S1:C1)\n"
            "- citations[].url must match the referenced source_id"
        )

        kept: list[str] = []
        for s in sources:
            cand = kept + [s]
            user_cand = "\n\n".join(
                [
                    f"USER_QUERY:\n{query}",
                    "SOURCES:\n" + "\n\n".join(cand),
                    task,
                    cite_fmt,
                ]
            )
            if max_prompt_chars and len(user_cand) > max_prompt_chars:
                break
            kept = cand

        user = "\n\n".join(
            [
                f"USER_QUERY:\n{query}",
                "SOURCES:\n" + "\n\n".join(kept),
                task,
                cite_fmt,
            ]
        )
        return [
            {
                "role": "system",
                "content": "You are a research assistant. Output JSON only.",
            },
            {"role": "user", "content": user},
        ]

    async def _build_overview(
        self,
        *,
        query: str,
        results: list[ResultItem],
        use_model: str,
        max_sources: int,
        max_chunks_per_source: int,
        max_chunk_chars: int,
        max_prompt_chars: int,
        max_output_tokens: int,
        self_heal_retries: int,
        force_language: str,
    ) -> OverviewResult:
        active_model = self.settings.llm.resolve_model(use_model)
        model_alias = active_model.name

        messages = self._build_messages(
            query=query,
            results=results,
            max_sources=max_sources,
            max_chunks_per_source=max_chunks_per_source,
            max_chunk_chars=max_chunk_chars,
            max_prompt_chars=max_prompt_chars,
            force_language=force_language,
        )
        schema = self._schema()
        retries = max(0, int(self_heal_retries))
        cur_messages = list(messages)

        with self.span("overview.build", model=model_alias) as sp:
            sp.set_attr("backend", str(active_model.backend))
            sp.set_attr("model_name", str(active_model.name))
            sp.set_attr("schema_strict", bool(active_model.schema_strict))
            sp.set_attr("self_heal_retries", int(retries))

            last_exc: Exception | None = None
            last_res: ChatJSONResult | None = None

            for attempt in range(retries + 1):
                sp.set_attr("attempt", int(attempt))
                try:
                    last_res = await self._llm.chat_json(
                        model=model_alias,
                        messages=cur_messages,
                        schema=schema,
                        timeout_s=float(active_model.timeout_s),
                    )
                    out = OverviewLLMOutput.model_validate(last_res.data)
                    out = self._sanitize_overview(out, results)
                    summary = self._truncate_summary(
                        out.summary,
                        max_output_tokens=max_output_tokens,
                    )
                    return OverviewResult(
                        summary=summary,
                        key_points=list(out.key_points or []),
                        citations=list(out.citations or []),
                        usage=LLMUsage(
                            prompt_tokens=int(last_res.usage.prompt_tokens),
                            completion_tokens=int(last_res.usage.completion_tokens),
                            total_tokens=int(last_res.usage.total_tokens),
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    last_exc = (
                        exc if isinstance(exc, Exception) else Exception(str(exc))
                    )
                    sp.add_event(
                        "overview.attempt_failed",
                        attempt=int(attempt),
                        error_type=type(exc).__name__,
                    )
                    if attempt < retries and _healable(exc):
                        cur_messages = cur_messages + [
                            {
                                "role": "user",
                                "content": (
                                    "Your previous output did not validate. "
                                    "Return JSON only that strictly matches the given schema. "
                                    "Do not include markdown or extra keys."
                                ),
                            }
                        ]
                        continue
                    raise

            assert last_exc is not None
            raise last_exc

    def _truncate_summary(self, summary: str, *, max_output_tokens: int) -> str:
        if max_output_tokens <= 0:
            return summary or ""
        return _truncate_to_token_budget(
            summary or "", max_tokens=int(max_output_tokens)
        )

    def _sanitize_overview(
        self, overview: OverviewLLMOutput, results: list[ResultItem]
    ) -> OverviewLLMOutput:
        sid_to_url: dict[str, str] = {}
        sid_to_chunks: dict[str, set[str]] = {}
        for r in results:
            sid = (r.source_id or "").strip()
            if not sid:
                continue
            if r.url:
                sid_to_url[sid] = r.url
            chunks = set()
            if r.page and r.page.chunks:
                for ch in r.page.chunks:
                    if ch.chunk_id:
                        chunks.add(ch.chunk_id)
            sid_to_chunks[sid] = chunks

        kept = []
        seen: set[tuple[str, str, str]] = set()
        for c in overview.citations or []:
            sid = (c.source_id or "").strip()
            if not sid or sid not in sid_to_chunks:
                continue

            url = sid_to_url.get(sid) or (c.url or "")
            chunk_id = c.chunk_id
            if chunk_id and chunk_id not in sid_to_chunks.get(sid, set()):
                chunk_id = None

            key = (sid, chunk_id or "", (c.quote or "").strip())
            if key in seen:
                continue
            seen.add(key)
            kept.append(
                c.model_copy(
                    update={
                        "url": url,
                        "chunk_id": chunk_id,
                    }
                )
            )

        renum = [
            c.model_copy(update={"cite_id": f"C{i}"}) for i, c in enumerate(kept, 1)
        ]
        return overview.model_copy(update={"citations": renum})

    def _build_source(self, ctx: FetchStepContext) -> ResultItem | None:
        if not (ctx.page.markdown or "").strip():
            return None
        chunks = list(ctx.page.chunks or [])
        if not chunks:
            excerpt = (ctx.extracted.plain_text if ctx.extracted else "").strip()
            if not excerpt:
                excerpt = (ctx.page.markdown or "").replace("\n", " ").strip()
            if excerpt:
                chunks = [PageChunk(chunk_id="S1:C1", text=excerpt[:1000], score=1.0)]

        source = ResultItem(
            source_id="S1",
            url=ctx.request.url,
            title=(ctx.extracted.title if ctx.extracted else "") or "",
            snippet=((ctx.extracted.plain_text if ctx.extracted else "") or "")[:260],
            score=1.0,
            page=ctx.page.model_copy(update={"chunks": chunks}),
        )
        for idx, ch in enumerate(source.page.chunks, 1):
            ch.chunk_id = ch.chunk_id or f"S1:C{idx}"
        return source


def _healable(exc: Exception) -> bool:
    return isinstance(exc, (ValidationError, TypeError, ValueError))


def _truncate_to_token_budget(text: str, *, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    if not text:
        return ""

    out_chars: list[str] = []
    tokens = 0
    i = 0
    n = len(text)

    def add_token(cnt: int) -> bool:
        nonlocal tokens
        if tokens + cnt > max_tokens:
            return False
        tokens += cnt
        return True

    while i < n:
        ch = text[i]
        if ch.isspace():
            out_chars.append(ch)
            i += 1
            continue

        if ch.isascii() and ch.isalnum():
            j = i + 1
            while j < n and text[j].isascii() and text[j].isalnum():
                j += 1
            run = text[i:j]
            need = (len(run) + 3) // 4
            if not add_token(need):
                break
            out_chars.append(run)
            i = j
            continue

        if _CJK_RE.fullmatch(ch):
            if not add_token(1):
                break
            out_chars.append(ch)
            i += 1
            continue

        if not add_token(1):
            break
        out_chars.append(ch)
        i += 1

    out = "".join(out_chars)
    if out == text:
        return out
    if out and tokens + 1 <= max_tokens:
        out = out + "..."
    return out


__all__ = ["FetchOverviewStep"]
