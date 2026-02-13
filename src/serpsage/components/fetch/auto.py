from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from typing_extensions import override

from serpsage.components.fetch.utils import (
    estimate_text_quality,
    has_spa_signals,
)
from serpsage.contracts.services import CacheBase, FetcherBase, RateLimiterBase
from serpsage.models.fetch import FetchAttempt, FetchResult
from serpsage.settings.models import RetrySettings
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.components.fetch.curl_cffi import CurlCffiFetcher
    from serpsage.components.fetch.http import HttpxFetcher
    from serpsage.components.fetch.playwright import PlaywrightFetcher
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.settings.models import FetchSettings

_MIN_BYTES = 128
_FETCH_MODES = {"httpx", "curl_cffi", "playwright"}
_CONTENT_KINDS = {"html", "pdf", "text", "binary", "unknown"}


def _hash_key(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def _encode_fetch_cache(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _decode_fetch_cache(blob: bytes) -> dict[str, Any]:
    return json.loads(blob.decode("utf-8"))


def _should_disable_cache(*, fetch_cfg: FetchSettings) -> bool:
    cookies = getattr(fetch_cfg, "cookies", None) or {}
    if cookies:
        return True
    extra = {
        str(k).lower(): str(v)
        for k, v in (getattr(fetch_cfg, "extra_headers", None) or {}).items()
    }
    return bool("cookie" in extra or "authorization" in extra)


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
        self.bind_deps(cache, rate_limiter, httpx_fetcher, curl_fetcher, playwright_fetcher)

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
        fetch_cfg = self.settings.enrich.fetch
        host = urlparse(url).netloc.lower()
        backend = str(fetch_cfg.backend or "auto").lower()
        depth_key = str(depth or "medium").lower()

        cache_allowed = not _should_disable_cache(fetch_cfg=fetch_cfg)
        cache_key = _hash_key(
            {
                "url": url,
                "kind": "fetch",
                "strategy": backend,
                "depth": depth_key,
                "allow_render": bool(allow_render),
                "rank_band": int(rank_index),
                "version": str(fetch_cfg.auto.version or "v2"),
            }
        )

        with self.span("fetch.auto", url=url, strategy=backend) as sp:
            sp.set_attr("depth", depth_key)
            sp.set_attr("rank_index", int(rank_index))
            sp.set_attr("allow_render", bool(allow_render))
            if cache_allowed:
                cached = await self._cache.aget(namespace="fetch", key=cache_key)
                if cached:
                    sp.set_attr("cache_hit", True)
                    payload = _decode_fetch_cache(cached)
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
                        headers={
                            str(k): str(v)
                            for k, v in (payload.get("headers") or {}).items()
                        },
                    )
            sp.set_attr("cache_hit", False)

            await self._rl.acquire(host=host)
            try:
                attempt = await self._fetch_useful(
                    url=url,
                    strategy=backend,
                    span=sp,
                    timeout_s=timeout_s,
                    allow_render=allow_render,
                    depth=depth_key,
                    rank_index=rank_index,
                )
            finally:
                await self._rl.release(host=host)

            if cache_allowed and not attempt.blocked:
                payload = {
                    "status_code": int(attempt.status_code),
                    "content_type": attempt.content_type,
                    "content_hex": attempt.content.hex(),
                    "url": attempt.url,
                    "fetch_mode": attempt.fetch_mode,
                    "rendered": bool(attempt.rendered),
                    "content_kind": attempt.content_kind,
                    "headers": dict(attempt.headers or {}),
                    "content_encoding": attempt.content_encoding,
                    "content_length_header": attempt.content_length_header,
                }
                await self._cache.aset(
                    namespace="fetch",
                    key=cache_key,
                    value=_encode_fetch_cache(payload),
                    ttl_s=int(self.settings.cache.fetch_ttl_s),
                )

            return FetchResult(
                url=attempt.url,
                status_code=int(attempt.status_code),
                content_type=attempt.content_type,
                content=attempt.content,
                fetch_mode=attempt.fetch_mode,
                rendered=bool(attempt.rendered),
                content_kind=attempt.content_kind,
                headers=dict(attempt.headers or {}),
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
            )
            if self._is_useful(res):
                return res
            raise RuntimeError("fetch_unusable:httpx")

        if strategy == "curl_cffi":
            if self._curl is None:
                raise RuntimeError(
                    "curl_cffi fetch strategy requested but curl fetcher not available"
                )
            res = await self._curl.fetch_attempt(url=url, span=span, timeout_s=timeout_s)
            if self._is_useful(res):
                return res
            raise RuntimeError("fetch_unusable:curl_cffi")

        if strategy == "playwright":
            if self._playwright is None:
                raise RuntimeError("playwright fetch strategy requested but unavailable")
            res = await self._playwright.fetch_attempt(
                url=url,
                span=span,
                timeout_s=timeout_s,
                render_reason="backend_playwright",
            )
            if self._is_useful(res):
                return res
            raise RuntimeError("fetch_unusable:playwright")

        # strategy=auto
        candidates: list[FetchAttempt] = []

        browser_res = await self._httpx.fetch_attempt(
            url=url,
            profile="browser",
            span=span,
            retry=RetrySettings(max_attempts=1),
            timeout_s=timeout_s,
        )
        candidates.append(browser_res)
        if self._is_useful(browser_res):
            return browser_res

        compat_res = await self._httpx.fetch_attempt(
            url=url,
            profile="compat",
            span=span,
            retry=RetrySettings(max_attempts=1),
            timeout_s=timeout_s,
        )
        candidates.append(compat_res)
        if self._is_useful(compat_res):
            return compat_res

        if self._curl is not None:
            curl_res = await self._curl.fetch_attempt(
                url=url,
                span=span,
                retry=RetrySettings(max_attempts=1),
                timeout_s=timeout_s,
            )
            candidates.append(curl_res)
            if self._is_useful(curl_res):
                return curl_res

        best_http = max(candidates, key=self._candidate_score)
        if self._should_render(best_http, allow_render=allow_render, depth=depth, rank_index=rank_index):
            render_reason = self._render_reason(best_http)
            span.set_attr("render_triggered", True)
            span.set_attr("render_reason", render_reason)
            if self._playwright is not None:
                render_res = await self._playwright.fetch_attempt(
                    url=url,
                    span=span,
                    timeout_s=timeout_s,
                    render_reason=render_reason,
                )
                candidates.append(render_res)
                if self._is_useful(render_res):
                    return render_res
        else:
            span.set_attr("render_triggered", False)

        best = max(candidates, key=self._candidate_score)
        if self._is_useful(best):
            return best
        raise RuntimeError("fetch_unusable:auto")

    def _is_blocked(self, res: FetchAttempt) -> bool:
        blocked_markers = self.settings.enrich.fetch.quality_gate.blocked_markers or []
        if int(res.status_code) in {401, 403}:
            return True
        if not res.content:
            return True
        try:
            sample = res.content[:16_384].decode("utf-8", errors="ignore")
        except Exception:
            return False
        marker_re = re.compile(
            "|".join(re.escape(str(m)) for m in blocked_markers if m),
            re.IGNORECASE,
        )
        return bool(marker_re.search(sample)) if blocked_markers else False

    def _is_useful(self, res: FetchAttempt) -> bool:
        status = int(res.status_code or 0)
        if status < 200 or status >= 400:
            return False
        if self._is_blocked(res):
            return False
        if len(res.content or b"") < _MIN_BYTES:
            return False
        if res.content_kind in {"binary", "unknown"}:
            return False
        qcfg = self.settings.enrich.fetch.quality_gate
        if res.content_kind == "html":
            if int(res.text_chars or 0) < int(qcfg.min_text_chars):
                return False
            if float(res.content_score or 0.0) < float(qcfg.min_content_score):
                return False
        return True

    def _candidate_score(self, res: FetchAttempt) -> float:
        status_bonus = 0.0
        st = int(res.status_code or 0)
        if 200 <= st < 300:
            status_bonus = 0.25
        elif 300 <= st < 400:
            status_bonus = 0.1
        blocked_penalty = 0.5 if self._is_blocked(res) else 0.0
        text_bonus = min(0.3, float(max(0, int(res.text_chars or 0))) / 8000.0)
        return float(res.content_score or 0.0) + status_bonus + text_bonus - blocked_penalty

    def _should_render(
        self,
        res: FetchAttempt,
        *,
        allow_render: bool,
        depth: str,
        rank_index: int,
    ) -> bool:
        if not allow_render or self._playwright is None:
            return False
        pw_cfg = self.settings.enrich.fetch.playwright
        if not bool(pw_cfg.enabled):
            return False
        if self._is_blocked(res):
            return True
        depth_limits = pw_cfg.max_render_pages_per_depth or {}
        depth_cap = int(depth_limits.get(depth, 0))
        if rank_index >= depth_cap:
            return False
        auto_cfg = self.settings.enrich.fetch.auto
        if not bool(auto_cfg.enable_speculative_render) and not self._is_blocked(res):
            return False
        if bool(auto_cfg.render_for_top_rank_only) and rank_index > 0 and depth == "low":
            return False

        qcfg = self.settings.enrich.fetch.quality_gate
        if res.content_kind != "html":
            return False
        if int(res.text_chars or 0) < int(qcfg.min_text_chars):
            return True
        if float(res.content_score or 0.0) < float(qcfg.min_content_score):
            return True
        _, _, script_ratio = estimate_text_quality(res.content, content_kind="html")
        if script_ratio >= float(qcfg.script_ratio_threshold) and has_spa_signals(res.content):
            return True
        return False

    def _render_reason(self, res: FetchAttempt) -> str:
        qcfg = self.settings.enrich.fetch.quality_gate
        if self._is_blocked(res):
            return "challenge_page"
        if int(res.text_chars or 0) < int(qcfg.min_text_chars):
            return "low_text_chars"
        if float(res.content_score or 0.0) < float(qcfg.min_content_score):
            return "low_content_score"
        _, _, script_ratio = estimate_text_quality(res.content, content_kind="html")
        if script_ratio >= float(qcfg.script_ratio_threshold) and has_spa_signals(res.content):
            return "spa_script_heavy"
        return "quality_gate"


__all__ = ["AutoFetcher"]
