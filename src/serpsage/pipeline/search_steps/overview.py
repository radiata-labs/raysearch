from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import httpx
from pydantic import ValidationError

from serpsage.app.response import OverviewLLMOutput, OverviewResult, ResultItem
from serpsage.components.overview.schema import overview_json_schema
from serpsage.models.errors import AppError
from serpsage.models.llm import LLMUsage
from serpsage.models.pipeline import SearchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import CacheBase, LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.models.llm import ChatJSONResult

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")


class SearchOverviewStep(PipelineStep[SearchStepContext]):
    span_name = "step.search_overview"

    def __init__(
        self, *, rt: Runtime, llm: LLMClientBase, cache: CacheBase
    ) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self._cache = cache
        self.bind_deps(llm, cache)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        profile = self.settings.search.overview
        enabled = profile.enabled_default
        if ctx.request.overview is not None:
            enabled = bool(ctx.request.overview)
        if not enabled or not ctx.results:
            return ctx

        llm_cfg = self.settings.llm.resolve_model(profile.use_model)
        messages = self._build_messages(
            query=ctx.request.query,
            results=ctx.results,
            max_sources=int(profile.max_sources),
            max_chunks_per_source=int(profile.max_chunks_per_source),
            max_chunk_chars=int(profile.max_chunk_chars),
            max_prompt_chars=int(profile.max_prompt_chars),
            force_language=str(profile.force_language),
        )
        schema = self._schema()
        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        span.set_attr("backend", str(llm_cfg.backend))
        span.set_attr("model_name", str(llm_cfg.name))
        span.set_attr("model", str(llm_cfg.model))
        span.set_attr("schema_strict", bool(llm_cfg.schema_strict))
        span.set_attr("prompt_chars", int(prompt_chars))
        span.set_attr("max_summary_tokens", int(profile.max_output_tokens))

        cache_ttl_s = int(profile.cache_ttl_s)
        cache_key: str | None = None
        if cache_ttl_s > 0:
            cache_key = self._overview_cache_key(
                use_model=str(profile.use_model),
                messages=messages,
                schema=schema,
                schema_strict=bool(llm_cfg.schema_strict),
            )
            cached = await self._cache.aget(namespace="overview:search", key=cache_key)
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
            overview = await self._build_overview(
                query=ctx.request.query,
                results=ctx.results,
                use_model=str(profile.use_model),
                max_sources=int(profile.max_sources),
                max_chunks_per_source=int(profile.max_chunks_per_source),
                max_chunk_chars=int(profile.max_chunk_chars),
                max_prompt_chars=int(profile.max_prompt_chars),
                max_output_tokens=int(profile.max_output_tokens),
                self_heal_retries=int(profile.self_heal_retries),
                force_language=str(profile.force_language),
            )
            ctx.overview = overview
            if cache_ttl_s > 0 and cache_key:
                await self._cache.aset(
                    namespace="overview:search",
                    key=cache_key,
                    value=overview.model_dump_json().encode("utf-8"),
                    ttl_s=cache_ttl_s,
                )
        except Exception as exc:  # noqa: BLE001
            retries = max(0, int(profile.self_heal_retries))
            code, details = self._map_overview_error(
                exc if isinstance(exc, Exception) else Exception(str(exc)),
                backend=str(llm_cfg.backend),
                model_name=str(llm_cfg.name),
                model=str(llm_cfg.model),
                base_url=str(llm_cfg.base_url),
                attempt=retries,
            )
            ctx.errors.append(AppError(code=code, message=str(exc), details=details))
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
        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        retries = max(0, int(self_heal_retries))
        cur_messages = list(messages)

        with self.span("overview.build", model=model_alias) as sp:
            sp.set_attr("backend", str(active_model.backend))
            sp.set_attr("model_name", str(active_model.name))
            sp.set_attr("schema_strict", bool(active_model.schema_strict))
            sp.set_attr("prompt_chars", int(prompt_chars))
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
        return _truncate_to_token_budget(summary or "", max_tokens=int(max_output_tokens))

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

    def _overview_cache_key(
        self,
        *,
        use_model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_strict: bool,
    ) -> str:
        payload = {
            "use_model": use_model,
            "messages": messages,
            "schema": schema,
            "schema_strict": bool(schema_strict),
        }
        return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()

    def _map_overview_error(
        self,
        exc: Exception,
        *,
        backend: str,
        model_name: str,
        model: str,
        base_url: str,
        attempt: int,
    ) -> tuple[str, dict[str, Any]]:
        details: dict[str, Any] = {
            "backend": backend,
            "model_name": model_name,
            "model": model,
            "base_url": base_url,
            "attempt": int(attempt),
            "type": type(exc).__name__,
        }

        request_id = getattr(exc, "request_id", None)
        if request_id:
            details["request_id"] = str(request_id)

        status = _extract_status_code(exc)
        if status is not None:
            details["status_code"] = int(status)

        code = "overview_failed"
        if _looks_like_timeout(exc, status=status):
            code = "overview_timeout"
        elif _looks_like_rate_limited(exc, status=status):
            code = "overview_rate_limited"
        elif _looks_like_auth_error(exc, status=status):
            code = "overview_auth_failed"
        elif status is not None and 500 <= int(status) < 600:
            code = "overview_server_error"
        elif status is not None and 400 <= int(status) < 500:
            code = "overview_bad_request"
        return code, details


def _healable(exc: Exception) -> bool:
    return isinstance(exc, (ValidationError, TypeError, ValueError))


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        raw = getattr(exc, attr, None)
        if raw is None:
            continue
        try:
            status = int(raw)
        except Exception:  # noqa: S112
            continue
        if 100 <= status <= 599:
            return status

    resp = getattr(exc, "response", None)
    raw_resp_status = getattr(resp, "status_code", None)
    if raw_resp_status is None:
        return None
    try:
        status = int(raw_resp_status)
    except Exception:  # noqa: BLE001
        return None
    if 100 <= status <= 599:
        return status
    return None


def _looks_like_timeout(exc: Exception, *, status: int | None) -> bool:
    if status == 408:
        return True
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name:
        return True
    msg = str(exc).lower()
    return (
        "timed out" in msg
        or "timeout" in msg
        or "connection error" in msg
        or "connection timed out" in msg
    )


def _looks_like_rate_limited(exc: Exception, *, status: int | None) -> bool:
    if status == 429:
        return True
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "too many requests" in msg


def _looks_like_auth_error(exc: Exception, *, status: int | None) -> bool:
    if status in {401, 403}:
        return True
    name = type(exc).__name__.lower()
    if "auth" in name:
        return True
    msg = str(exc).lower()
    return "authentication" in msg or "permission denied" in msg


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


__all__ = ["SearchOverviewStep"]
