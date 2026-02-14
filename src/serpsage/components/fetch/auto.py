from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import urlparse

import anyio

from serpsage.components.fetch.utils import (
    blocked_marker_hit,
    estimate_text_quality,
    has_spa_signals,
)
from serpsage.contracts.services import FetcherBase, RateLimiterBase
from serpsage.models.fetch import FetchAttempt, FetchResult
from serpsage.settings.models import RetrySettings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from serpsage.components.fetch.curl_cffi import CurlCffiFetcher
    from serpsage.components.fetch.http import HttpxFetcher
    from serpsage.components.fetch.playwright import PlaywrightFetcher
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime

_MIN_BYTES = 96


class AutoFetcher(FetcherBase):
    def __init__(
        self,
        *,
        rt: Runtime,
        rate_limiter: RateLimiterBase,
        httpx_fetcher: HttpxFetcher,
        curl_fetcher: CurlCffiFetcher | None,
        playwright_fetcher: PlaywrightFetcher | None,
    ) -> None:
        super().__init__(rt=rt)
        self._rl = rate_limiter
        self._httpx = httpx_fetcher
        self._curl = curl_fetcher
        self._playwright = playwright_fetcher
        self.bind_deps(
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
        rank_index: int = 0,
    ) -> FetchResult:
        backend = str(self.settings.fetch.backend or "auto").lower()
        with self.span("fetch.auto", url=url, strategy=backend) as sp:
            sp.set_attr("rank_index", int(rank_index))
            sp.set_attr("allow_render", bool(allow_render))
            return await self._afetch_uncached(
                url=url,
                timeout_s=timeout_s,
                allow_render=allow_render,
                rank_index=rank_index,
                span=sp,
            )

    async def _afetch_uncached(
        self,
        *,
        url: str,
        timeout_s: float | None,
        allow_render: bool,
        rank_index: int,
        span: SpanBase,
    ) -> FetchResult:
        host = urlparse(url).netloc.lower()
        await self._rl.acquire(host=host)
        try:
            attempt = await self._fetch_useful(
                url=url,
                strategy=str(self.settings.fetch.backend or "auto").lower(),
                span=span,
                timeout_s=timeout_s,
                allow_render=allow_render,
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

    async def _fetch_useful(
        self,
        *,
        url: str,
        strategy: str,
        span: SpanBase,
        timeout_s: float | None,
        allow_render: bool,
        rank_index: int,
    ) -> FetchAttempt:
        if strategy == "httpx":
            res = await self._httpx.fetch_attempt(
                url=url,
                profile="browser",
                span=span,
                timeout_s=timeout_s,
            )
            if self._is_useful(res):
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
            )
            if self._is_useful(res):
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
            )
            if self._is_useful(res):
                return res
            raise RuntimeError("fetch_unusable:playwright")
        return await self._fetch_auto(
            url=url,
            span=span,
            timeout_s=timeout_s,
            allow_render=allow_render,
            rank_index=rank_index,
        )

    async def _fetch_auto(
        self,
        *,
        url: str,
        span: SpanBase,
        timeout_s: float | None,
        allow_render: bool,
        rank_index: int,
    ) -> FetchAttempt:
        base_candidates: list[FetchAttempt] = []
        winner: FetchAttempt | None = None
        render_launched = False
        span.set_attr("render_triggered", False)

        hedge_compat_s = 0.08
        hedge_curl_s = 0.11
        retry = RetrySettings(max_attempts=1, delay_ms=90)

        def make_runner(
            func: Callable[..., Awaitable[Any]], *args: object, **kwargs: object
        ) -> Callable[[SpanBase], Awaitable[FetchAttempt]]:
            async def runner(candidate_span: SpanBase) -> FetchAttempt:
                return await func(*args, span=candidate_span, **kwargs)

            return runner

        async with anyio.create_task_group() as tg:

            async def worker(
                label: str,
                delay_s: float,
                runner: Callable[[SpanBase], Awaitable[FetchAttempt]],
                can_trigger_render: bool = True,
            ) -> None:
                nonlocal winner, render_launched
                if delay_s > 0:
                    await anyio.sleep(delay_s)
                if winner is not None:
                    return
                with self.span("fetch.auto.candidate", url=url, mode=label) as csp:
                    try:
                        attempt = await runner(csp)
                    except Exception as exc:  # noqa: BLE001
                        attempt = self._failed_attempt(
                            url=url,
                            mode=label,
                            error=type(exc).__name__,
                        )
                        csp.set_attr("candidate_error_type", type(exc).__name__)

                    blocked = self._is_blocked(attempt)
                    useful = self._is_useful(attempt)
                    self._set_candidate_span_attrs(
                        span=csp,
                        attempt=attempt,
                        blocked=blocked,
                        useful=useful,
                    )
                    base_candidates.append(attempt)
                    span.set_attr("candidate_count", int(len(base_candidates)))

                    if can_trigger_render and self._should_render(
                        attempt,
                        allow_render=allow_render,
                        rank_index=rank_index,
                        render_launched=render_launched,
                    ):
                        render_launched = True
                        reason = self._render_reason(attempt)
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
                                    timeout_s=timeout_s,
                                    render_reason=reason,
                                ),
                                False,
                            )
                    if useful:
                        winner = attempt
                        self._set_winner_attrs(span=span, winner=attempt)
                        tg.cancel_scope.cancel()

            tg.start_soon(
                worker,
                "httpx:browser",
                0.0,
                make_runner(
                    self._httpx.fetch_attempt,
                    url=url,
                    profile="browser",
                    retry=retry,
                    timeout_s=timeout_s,
                ),
            )
            tg.start_soon(
                worker,
                "httpx:compat",
                hedge_compat_s,
                make_runner(
                    self._httpx.fetch_attempt,
                    url=url,
                    profile="compat",
                    retry=retry,
                    timeout_s=timeout_s,
                ),
            )
            if self._curl is not None:
                tg.start_soon(
                    worker,
                    "curl_cffi",
                    hedge_curl_s,
                    make_runner(
                        self._curl.fetch_attempt,
                        url=url,
                        retry=retry,
                        timeout_s=timeout_s,
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
                rank_index=rank_index,
                render_launched=False,
            ):
                reason = self._render_reason(best_http)
                span.set_attr("render_triggered", True)
                span.set_attr("render_reason", reason)
                try:
                    with self.span(
                        "fetch.auto.candidate",
                        url=url,
                        mode="playwright",
                    ) as csp:
                        render_attempt = await self._playwright.fetch_attempt(
                            url=url,
                            span=csp,
                            timeout_s=timeout_s,
                            render_reason=reason,
                        )
                        blocked = self._is_blocked(render_attempt)
                        useful = self._is_useful(render_attempt)
                        self._set_candidate_span_attrs(
                            span=csp,
                            attempt=render_attempt,
                            blocked=blocked,
                            useful=useful,
                        )
                        base_candidates.append(render_attempt)
                        span.set_attr("candidate_count", int(len(base_candidates)))
                        if useful:
                            winner = render_attempt
                            self._set_winner_attrs(span=span, winner=render_attempt)
                except Exception as exc:  # noqa: BLE001
                    span.set_attr("playwright_error", type(exc).__name__)

        if winner is not None:
            return self._attach_attempt_chain(winner, base_candidates)

        if base_candidates:
            best = max(base_candidates, key=self._candidate_score)
            if self._is_useful(best):
                self._set_winner_attrs(span=span, winner=best)
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
        marker_hit = self._has_block_marker(res=res)
        if not marker_hit:
            return False
        return self._is_low_quality_for_block(res)

    def _has_block_marker(self, *, res: FetchAttempt) -> bool:
        markers = tuple(self.settings.fetch.quality.blocked_markers or [])
        if not markers:
            return bool(res.blocked)
        if blocked_marker_hit(res.content, markers=markers):
            return True
        return bool(res.blocked)

    def _is_low_quality_for_block(self, res: FetchAttempt) -> bool:
        quality = self.settings.fetch.quality
        if len(res.content or b"") < _MIN_BYTES:
            return True
        if res.content_kind in {"binary", "unknown"}:
            return True
        if res.content_kind == "html":
            if int(res.text_chars or 0) < int(quality.min_text_chars):
                return True
            if float(res.content_score or 0.0) < float(quality.min_content_score):
                return True
        return False

    def _is_useful(self, res: FetchAttempt) -> bool:
        quality = self.settings.fetch.quality
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
            if int(res.text_chars or 0) < int(quality.min_text_chars):
                return False
            if float(res.content_score or 0.0) < float(quality.min_content_score):
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
        rank_index: int,
        render_launched: bool,
    ) -> bool:
        if not allow_render or render_launched:
            return False
        if not bool(self.settings.fetch.render.enabled) or self._playwright is None:
            return False

        max_pages = int(self.settings.fetch.quality.max_render_pages_search)
        if rank_index >= max_pages:
            return False
        if self._is_blocked(res):
            return True
        if int(res.status_code or 0) in {401, 403}:
            return True
        if res.content_kind != "html":
            return False

        quality = self.settings.fetch.quality
        if int(res.text_chars or 0) < int(quality.min_text_chars):
            return True
        if float(res.content_score or 0.0) < float(quality.min_content_score):
            return True
        _, _, script_ratio = estimate_text_quality(res.content, content_kind="html")
        return bool(
            script_ratio >= float(quality.script_ratio_threshold)
            and has_spa_signals(res.content)
        )

    def _render_reason(self, res: FetchAttempt) -> str:
        quality = self.settings.fetch.quality
        if self._is_blocked(res):
            return "challenge_page"
        if int(res.status_code or 0) in {401, 403}:
            return "status_forbidden"
        if int(res.text_chars or 0) < int(quality.min_text_chars):
            return "low_text_chars"
        if float(res.content_score or 0.0) < float(quality.min_content_score):
            return "low_content_score"
        _, _, script_ratio = estimate_text_quality(res.content, content_kind="html")
        if script_ratio >= float(quality.script_ratio_threshold) and has_spa_signals(
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

    def _set_candidate_span_attrs(
        self,
        *,
        span: SpanBase,
        attempt: FetchAttempt,
        blocked: bool,
        useful: bool,
    ) -> None:
        span.set_attr("status", int(attempt.status_code or 0))
        span.set_attr("fetch_mode", str(attempt.fetch_mode))
        span.set_attr("content_kind", str(attempt.content_kind))
        span.set_attr("content_bytes", int(len(attempt.content or b"")))
        span.set_attr("text_chars", int(attempt.text_chars or 0))
        span.set_attr("content_score", float(attempt.content_score or 0.0))
        span.set_attr("blocked", bool(blocked))
        span.set_attr("useful", bool(useful))
        if attempt.render_reason:
            span.set_attr("render_reason", str(attempt.render_reason))

    def _set_winner_attrs(self, *, span: SpanBase, winner: FetchAttempt) -> None:
        span.set_attr("winner_mode", str(winner.fetch_mode))
        span.set_attr("winner_status", int(winner.status_code or 0))
        span.set_attr("winner_content_kind", str(winner.content_kind))
        span.set_attr("winner_text_chars", int(winner.text_chars or 0))
        span.set_attr("winner_content_score", float(winner.content_score or 0.0))


__all__ = ["AutoFetcher"]
