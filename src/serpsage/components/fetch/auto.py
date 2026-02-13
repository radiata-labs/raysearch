from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import urlparse

import anyio

from serpsage.components.fetch.utils import (
    blocked_marker_hit,
    estimate_text_quality,
    has_spa_signals,
)
from serpsage.contracts.services import CacheBase, FetcherBase, RateLimiterBase
from serpsage.core.tuning import (
    DEFAULT_FETCH_VERSION,
    FETCH_BLOCKED_MARKERS,
    PLAYWRIGHT_ENABLED,
    fetch_profile_for_depth,
    normalize_depth,
)
from serpsage.models.fetch import FetchAttempt, FetchResult
from serpsage.settings.models import RetrySettings
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from serpsage.components.fetch.curl_cffi import CurlCffiFetcher
    from serpsage.components.fetch.http import HttpxFetcher
    from serpsage.components.fetch.playwright import PlaywrightFetcher
    from serpsage.contracts.lifecycle import SpanBase

_MIN_BYTES = 96
_FETCH_MODES = {"httpx", "curl_cffi", "playwright"}
_CONTENT_KINDS = {"html", "pdf", "text", "binary", "unknown"}
_CHALLENGE_RE = re.compile(
    "|".join(re.escape(x) for x in FETCH_BLOCKED_MARKERS),
    re.IGNORECASE,
)


def _hash_key(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def _encode_fetch_cache(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_fetch_cache(blob: bytes) -> dict[str, Any]:
    return json.loads(blob.decode("utf-8"))


@dataclass(slots=True)
class _Inflight:
    event: anyio.Event
    result: FetchResult | None = None
    error: Exception | None = None


class AutoFetcher(FetcherBase):
    def __init__(
        self,
        *,
        rt,
        cache: CacheBase,
        rate_limiter: RateLimiterBase,
        httpx_fetcher: HttpxFetcher,
        curl_fetcher: CurlCffiFetcher | None,
        playwright_fetcher: PlaywrightFetcher | None,
    ) -> None:
        super().__init__(rt=rt)
        self._cache = cache
        self._rl = rate_limiter
        self._httpx = httpx_fetcher
        self._curl = curl_fetcher
        self._playwright = playwright_fetcher
        self._inflight: dict[str, _Inflight] = {}
        self._inflight_lock = anyio.Lock()
        self.bind_deps(
            cache,
            rate_limiter,
            httpx_fetcher,
            curl_fetcher,
            playwright_fetcher,
        )

    @override
    async def afetch(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
        allow_render: bool = True,
        depth: str | None = None,
        rank_index: int = 0,
    ) -> FetchResult:
        depth_key = normalize_depth(depth)
        backend = str(self.settings.enrich.fetch.backend or "auto").lower()
        cache_key = _hash_key(
            {
                "url": url,
                "kind": "fetch",
                "strategy": backend,
                "depth": depth_key,
                "allow_render": bool(allow_render),
                "rank_band": int(rank_index),
                "version": str(DEFAULT_FETCH_VERSION),
            }
        )
        inflight_key = _hash_key(
            {
                "url": url,
                "depth": depth_key,
                "allow_render": bool(allow_render),
                "backend": backend,
                "timeout_s": timeout_s,
            }
        )
        with self.span("fetch.auto", url=url, strategy=backend) as sp:
            sp.set_attr("depth", depth_key)
            sp.set_attr("rank_index", int(rank_index))
            sp.set_attr("allow_render", bool(allow_render))
            cached = await self._cache.aget(namespace="fetch", key=cache_key)
            if cached:
                sp.set_attr("cache_hit", True)
                payload = _decode_fetch_cache(cached)
                return self._payload_to_result(url=url, payload=payload)
            sp.set_attr("cache_hit", False)
            result, is_leader = await self._run_inflight(
                key=inflight_key,
                runner=lambda: self._afetch_uncached(
                    url=url,
                    timeout_s=timeout_s,
                    allow_render=allow_render,
                    depth=depth_key,
                    rank_index=rank_index,
                    span=sp,
                ),
            )
            if is_leader:
                await self._cache.aset(
                    namespace="fetch",
                    key=cache_key,
                    value=_encode_fetch_cache(
                        {
                            "status_code": int(result.status_code),
                            "content_type": result.content_type,
                            "content_hex": result.content.hex(),
                            "url": result.url,
                            "fetch_mode": result.fetch_mode,
                            "rendered": bool(result.rendered),
                            "content_kind": result.content_kind,
                            "headers": dict(result.headers or {}),
                            "attempt_chain": list(result.attempt_chain or []),
                            "quality_score": float(result.quality_score or 0.0),
                        }
                    ),
                    ttl_s=int(self.settings.cache.fetch_ttl_s),
                )
            return result

    async def _run_inflight(
        self,
        *,
        key: str,
        runner,  # noqa: ANN001
    ) -> tuple[FetchResult, bool]:
        async with self._inflight_lock:
            existing = self._inflight.get(key)
            if existing is None:
                existing = _Inflight(event=anyio.Event())
                self._inflight[key] = existing
                leader = True
            else:
                leader = False
        if not leader:
            await existing.event.wait()
            if existing.error is not None:
                raise existing.error
            if existing.result is None:
                raise RuntimeError("fetch inflight failed without result")
            return existing.result, False

        try:
            result = await runner()
            existing.result = result
            return result, True
        except Exception as exc:  # noqa: BLE001
            existing.error = exc
            raise
        finally:
            existing.event.set()
            async with self._inflight_lock:
                self._inflight.pop(key, None)

    async def _afetch_uncached(
        self,
        *,
        url: str,
        timeout_s: float | None,
        allow_render: bool,
        depth: str,
        rank_index: int,
        span: SpanBase,
    ) -> FetchResult:
        host = urlparse(url).netloc.lower()
        await self._rl.acquire(host=host)
        try:
            attempt = await self._fetch_useful(
                url=url,
                strategy=str(self.settings.enrich.fetch.backend or "auto").lower(),
                span=span,
                timeout_s=timeout_s,
                allow_render=allow_render,
                depth=depth,
                rank_index=rank_index,
            )
        finally:
            await self._rl.release(host=host)
        return FetchResult(
            url=attempt.url,
            status_code=int(attempt.status_code),
            content_type=attempt.content_type,
            content=attempt.content,
            fetch_mode=attempt.fetch_mode,
            rendered=bool(attempt.rendered),
            content_kind=attempt.content_kind,
            headers=dict(attempt.headers or {}),
            attempt_chain=list(attempt.attempt_chain or []),
            quality_score=float(attempt.quality_score or attempt.content_score),
        )

    def _payload_to_result(self, *, url: str, payload: dict[str, Any]) -> FetchResult:
        fetch_mode = str(payload.get("fetch_mode") or "httpx")
        if fetch_mode not in _FETCH_MODES:
            fetch_mode = "httpx"
        content_kind = str(payload.get("content_kind") or "unknown")
        if content_kind not in _CONTENT_KINDS:
            content_kind = "unknown"
        return FetchResult(
            url=str(payload.get("url") or url),
            status_code=int(payload["status_code"]),
            content_type=payload.get("content_type"),
            content=bytes.fromhex(payload["content_hex"]),
            fetch_mode=fetch_mode,  # type: ignore[arg-type]
            rendered=bool(payload.get("rendered", False)),
            content_kind=content_kind,  # type: ignore[arg-type]
            headers={str(k): str(v) for k, v in (payload.get("headers") or {}).items()},
            attempt_chain=[
                str(x) for x in (payload.get("attempt_chain") or []) if str(x).strip()
            ],
            quality_score=float(payload.get("quality_score") or 0.0),
        )

    async def _fetch_useful(
        self,
        *,
        url: str,
        strategy: str,
        span: SpanBase,
        timeout_s: float | None,
        allow_render: bool,
        depth: str,
        rank_index: int,
    ) -> FetchAttempt:
        if strategy == "httpx":
            res = await self._httpx.fetch_attempt(
                url=url,
                profile="browser",
                span=span,
                timeout_s=timeout_s,
                depth=depth,
            )
            if self._is_useful(res, depth=depth):
                return res
            raise RuntimeError("fetch_unusable:httpx")
        if strategy == "curl_cffi":
            if self._curl is None:
                raise RuntimeError(
                    "curl_cffi fetch strategy requested but curl fetcher not available"
                )
            res = await self._curl.fetch_attempt(
                url=url,
                span=span,
                timeout_s=timeout_s,
                depth=depth,
            )
            if self._is_useful(res, depth=depth):
                return res
            raise RuntimeError("fetch_unusable:curl_cffi")
        if strategy == "playwright":
            if self._playwright is None:
                raise RuntimeError(
                    "playwright fetch strategy requested but unavailable"
                )
            res = await self._playwright.fetch_attempt(
                url=url,
                span=span,
                timeout_s=timeout_s,
                render_reason="backend_playwright",
                depth=depth,
            )
            if self._is_useful(res, depth=depth):
                return res
            raise RuntimeError("fetch_unusable:playwright")
        return await self._fetch_auto(
            url=url,
            span=span,
            timeout_s=timeout_s,
            allow_render=allow_render,
            depth=depth,
            rank_index=rank_index,
        )

    async def _fetch_auto(
        self,
        *,
        url: str,
        span: SpanBase,
        timeout_s: float | None,
        allow_render: bool,
        depth: str,
        rank_index: int,
    ) -> FetchAttempt:
        tune = fetch_profile_for_depth(depth)
        base_candidates: list[FetchAttempt] = []
        winner: FetchAttempt | None = None
        render_launched = False

        def make_runner(func: Callable[..., Awaitable[Any]], *args, **kwargs):
            async def runner():
                return await func(*args, **kwargs)

            return runner

        async with anyio.create_task_group() as tg:

            async def worker(
                label: str,
                delay_s: float,
                runner: Callable[..., Awaitable[Any]],
                can_trigger_render: bool = True,
            ) -> None:  # noqa: ANN001
                nonlocal winner, render_launched
                if delay_s > 0:
                    await anyio.sleep(delay_s)
                if winner is not None:
                    return
                try:
                    attempt = await runner()
                except Exception as exc:  # noqa: BLE001
                    attempt = self._failed_attempt(
                        url=url,
                        mode=label,
                        error=type(exc).__name__,
                    )
                base_candidates.append(attempt)
                if can_trigger_render and self._should_render(
                    attempt,
                    allow_render=allow_render,
                    depth=depth,
                    rank_index=rank_index,
                    render_launched=render_launched,
                ):
                    render_launched = True
                    reason = self._render_reason(attempt, depth=depth)
                    span.set_attr("render_triggered", True)
                    span.set_attr("render_reason", reason)
                    if self._playwright is not None:
                        tg.start_soon(
                            worker,
                            "playwright",
                            0.0,
                            make_runner(
                                self._playwright.fetch_attempt,
                                url=url,
                                span=span,
                                timeout_s=timeout_s,
                                render_reason=reason,
                                depth=depth,
                            ),
                            False,
                        )
                if self._is_useful(attempt, depth=depth):
                    winner = attempt
                    tg.cancel_scope.cancel()

            tg.start_soon(
                worker,
                "httpx:browser",
                0.0,
                lambda: self._httpx.fetch_attempt(
                    url=url,
                    profile="browser",
                    span=span,
                    retry=RetrySettings(
                        max_attempts=1, delay_ms=int(tune.retry_delay_ms)
                    ),
                    timeout_s=timeout_s,
                    depth=depth,
                ),
            )
            tg.start_soon(
                worker,
                "httpx:compat",
                float(tune.hedge_delay_ms) / 2000.0,
                lambda: self._httpx.fetch_attempt(
                    url=url,
                    profile="compat",
                    span=span,
                    retry=RetrySettings(
                        max_attempts=1, delay_ms=int(tune.retry_delay_ms)
                    ),
                    timeout_s=timeout_s,
                    depth=depth,
                ),
            )
            if self._curl is not None:
                tg.start_soon(
                    worker,
                    "curl_cffi",
                    float(tune.hedge_delay_ms) / 1000.0,
                    make_runner(
                        self._curl.fetch_attempt,
                        url=url,
                        span=span,
                        retry=RetrySettings(
                            max_attempts=1,
                            delay_ms=int(tune.retry_delay_ms),
                        ),
                        timeout_s=timeout_s,
                        depth=depth,
                    ),
                )

        if winner is None and self._playwright is not None and not render_launched:
            best_http = (
                max(base_candidates, key=self._candidate_score)
                if base_candidates
                else None
            )
            if best_http is not None and self._should_render(
                best_http,
                allow_render=allow_render,
                depth=depth,
                rank_index=rank_index,
                render_launched=False,
            ):
                reason = self._render_reason(best_http, depth=depth)
                span.set_attr("render_triggered", True)
                span.set_attr("render_reason", reason)
                try:
                    render_attempt = await self._playwright.fetch_attempt(
                        url=url,
                        span=span,
                        timeout_s=timeout_s,
                        render_reason=reason,
                        depth=depth,
                    )
                    base_candidates.append(render_attempt)
                    if self._is_useful(render_attempt, depth=depth):
                        winner = render_attempt
                except Exception as exc:  # noqa: BLE001
                    span.set_attr("playwright_error", type(exc).__name__)
            else:
                span.set_attr("render_triggered", False)

        if winner is not None:
            return self._attach_attempt_chain(winner, base_candidates)

        if base_candidates:
            best = max(base_candidates, key=self._candidate_score)
            if self._is_useful(best, depth=depth):
                return self._attach_attempt_chain(best, base_candidates)
            best = self._attach_attempt_chain(best, base_candidates)
            raise RuntimeError(
                f"fetch_unusable:auto:{best.fetch_mode}:{best.status_code}"
            )
        raise RuntimeError("fetch_unusable:auto:no_candidates")

    def _is_blocked(self, res: FetchAttempt) -> bool:
        status = int(res.status_code or 0)
        if status in {401, 403}:
            return True
        if blocked_marker_hit(res.content):
            return True
        if res.content and _CHALLENGE_RE.search(
            res.content[:18_000].decode("utf-8", errors="ignore")
        ):
            return True
        return bool(res.blocked)

    def _is_useful(self, res: FetchAttempt, *, depth: str) -> bool:
        tune = fetch_profile_for_depth(depth)
        status = int(res.status_code or 0)
        if status < 200 or status >= 400:
            return False
        if self._is_blocked(res):
            return False
        if len(res.content or b"") < _MIN_BYTES:
            return False
        if res.content_kind in {"binary", "unknown"}:
            return False
        if res.content_kind == "html":
            if int(res.text_chars or 0) < int(tune.min_text_chars):
                return False
            if float(res.content_score or 0.0) < float(tune.min_content_score):
                return False
        return True

    def _candidate_score(self, res: FetchAttempt) -> float:
        status = int(res.status_code or 0)
        status_bonus = 0.0
        if 200 <= status < 300:
            status_bonus = 0.30
        elif 300 <= status < 400:
            status_bonus = 0.12
        mode_bonus = 0.10 if res.rendered else 0.0
        text_bonus = min(0.35, float(max(0, int(res.text_chars or 0))) / 8000.0)
        blocked_penalty = 0.60 if self._is_blocked(res) else 0.0
        return (
            float(res.quality_score or res.content_score or 0.0)
            + status_bonus
            + mode_bonus
            + text_bonus
            - blocked_penalty
        )

    def _should_render(
        self,
        res: FetchAttempt,
        *,
        allow_render: bool,
        depth: str,
        rank_index: int,
        render_launched: bool,
    ) -> bool:
        if not allow_render or render_launched:
            return False
        if not bool(PLAYWRIGHT_ENABLED) or self._playwright is None:
            return False
        tune = fetch_profile_for_depth(depth)
        if rank_index >= int(tune.max_render_pages):
            return False
        if self._is_blocked(res):
            return True
        status = int(res.status_code or 0)
        if status in {401, 403}:
            return True
        if res.content_kind != "html":
            return False
        if int(res.text_chars or 0) < int(tune.min_text_chars):
            return True
        if float(res.content_score or 0.0) < float(tune.min_content_score):
            return True
        _, _, script_ratio = estimate_text_quality(res.content, content_kind="html")
        return bool(
            script_ratio >= float(tune.script_ratio_threshold)
            and has_spa_signals(res.content)
            and bool(tune.allow_speculative_render)
        )

    def _render_reason(self, res: FetchAttempt, *, depth: str) -> str:
        tune = fetch_profile_for_depth(depth)
        if self._is_blocked(res):
            return "challenge_page"
        if int(res.status_code or 0) in {401, 403}:
            return "status_forbidden"
        if int(res.text_chars or 0) < int(tune.min_text_chars):
            return "low_text_chars"
        if float(res.content_score or 0.0) < float(tune.min_content_score):
            return "low_content_score"
        _, _, script_ratio = estimate_text_quality(res.content, content_kind="html")
        if script_ratio >= float(tune.script_ratio_threshold) and has_spa_signals(
            res.content
        ):
            return "spa_script_heavy"
        return "quality_gate"

    def _failed_attempt(self, *, url: str, mode: str, error: str) -> FetchAttempt:
        fetch_mode = "httpx"
        if mode.startswith("curl"):
            fetch_mode = "curl_cffi"
        elif mode.startswith("playwright"):
            fetch_mode = "playwright"
        return FetchAttempt(
            url=url,
            status_code=0,
            content_type=None,
            content=b"",
            strategy_used=fetch_mode,  # type: ignore[arg-type]
            fetch_mode=fetch_mode,  # type: ignore[arg-type]
            rendered=bool(fetch_mode == "playwright"),
            content_kind="unknown",
            headers={},
            content_encoding=None,
            content_length_header=None,
            content_score=0.0,
            text_chars=0,
            blocked=False,
            render_reason=None,
            attempt_chain=[f"{mode}:error:{error}"],
            quality_score=0.0,
        )

    def _attach_attempt_chain(
        self,
        winner: FetchAttempt,
        candidates: list[FetchAttempt],
    ) -> FetchAttempt:
        chain: list[str] = []
        for c in candidates:
            for item in c.attempt_chain or [c.fetch_mode]:
                if item not in chain:
                    chain.append(item)
        if not chain:
            chain = [winner.fetch_mode]
        return winner.model_copy(
            update={
                "attempt_chain": chain,
                "quality_score": float(max(0.0, self._candidate_score(winner))),
            }
        )


__all__ = ["AutoFetcher"]
