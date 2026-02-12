from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import httpx

from serpsage.app.response import OverviewResult
from serpsage.models.errors import AppError
from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import CacheBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.overview import OverviewBuilder


class OverviewStep(StepBase):
    span_name = "step.overview"

    def __init__(
        self, *, rt: Runtime, builder: OverviewBuilder, cache: CacheBase
    ) -> None:
        super().__init__(rt=rt)
        self._builder = builder
        self._cache = cache
        self.bind_deps(builder, cache)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        enabled = self.settings.overview.enabled
        if ctx.request.overview is not None:
            enabled = bool(ctx.request.overview)
        if not enabled:
            return ctx
        if not ctx.results:
            return ctx

        llm_cfg = self.settings.overview.resolve_model()
        model = llm_cfg.model
        messages = self._builder.build_messages(
            query=ctx.request.query, results=ctx.results
        )
        schema = self._builder.schema()

        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        span.set_attr("backend", str(llm_cfg.backend))
        span.set_attr("model_name", str(llm_cfg.name))
        span.set_attr("model", model)
        span.set_attr("schema_strict", bool(llm_cfg.schema_strict))
        span.set_attr("prompt_chars", int(prompt_chars))
        span.set_attr(
            "max_summary_tokens", int(self.settings.overview.max_output_tokens)
        )

        cache_ttl_s = int(self.settings.overview.cache_ttl_s)
        cache_key: str | None = None
        if cache_ttl_s > 0:
            cache_key = self._overview_cache_key(
                model=model,
                backend=str(llm_cfg.backend),
                model_name=str(llm_cfg.name),
                base_url=str(llm_cfg.base_url),
                messages=messages,
                schema=schema,
                schema_strict=bool(llm_cfg.schema_strict),
            )
            cached = await self._cache.aget(namespace="overview", key=cache_key)
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
            overview = await self._builder.build_overview(
                query=ctx.request.query, results=ctx.results
            )
            ctx.overview = overview

            if cache_ttl_s > 0 and cache_key:
                await self._cache.aset(
                    namespace="overview",
                    key=cache_key,
                    value=overview.model_dump_json().encode("utf-8"),
                    ttl_s=cache_ttl_s,
                )
        except Exception as exc:  # noqa: BLE001
            retries = max(0, int(self.settings.overview.self_heal_retries))
            code, details = self._map_overview_error(
                exc if isinstance(exc, Exception) else Exception(str(exc)),
                backend=str(llm_cfg.backend),
                model_name=str(llm_cfg.name),
                model=model,
                base_url=str(llm_cfg.base_url),
                attempt=retries,
            )
            ctx.errors.append(AppError(code=code, message=str(exc), details=details))
        return ctx

    def _overview_cache_key(
        self,
        model: str,
        backend: str,
        model_name: str,
        base_url: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_strict: bool,
    ) -> str:
        payload = {
            "backend": backend,
            "model_name": model_name,
            "model": model,
            "base_url": base_url,
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


__all__ = ["OverviewStep"]
