from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import urlparse

from serpsage.contracts.services import CacheBase, ExtractorBase, FetcherBase
from serpsage.models.fetch import FetchAttempt, FetchResult
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.components.fetch.curl_cffi import CurlCffiFetcher
    from serpsage.components.fetch.http import HttpxFetcher
    from serpsage.components.fetch.rate_limit import RateLimiter
    from serpsage.contracts.lifecycle import SpanBase


_BLOCKED_RE = re.compile(
    r"(cloudflare|cf-ray|cf-chl|just a moment|attention required|captcha|"
    r"verify you are human|access denied|please enable javascript|人机验证|访问受限)",
    re.IGNORECASE,
)
_BLOCKED_SCAN_BYTES = 16_384
_MIN_USEFUL_BYTES = 128


def _hash_key(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def _encode_fetch_cache(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _decode_fetch_cache(blob: bytes) -> dict[str, Any]:
    return json.loads(blob.decode("utf-8"))


def _should_disable_cache(*, fetch_cfg: Any, http_cfg: Any) -> bool:
    if getattr(http_cfg, "proxy", None):
        return True
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
        rate_limiter: RateLimiter,
        httpx_fetcher: HttpxFetcher,
        curl_fetcher: CurlCffiFetcher | None,
        extractor: ExtractorBase,
    ) -> None:
        super().__init__(rt=rt)
        self._cache = cache
        self._rl = rate_limiter
        self._httpx = httpx_fetcher
        self._curl = curl_fetcher
        self._extractor = extractor
        self.bind_deps(cache, rate_limiter, httpx_fetcher, curl_fetcher, extractor)

    @override
    async def afetch(self, *, url: str) -> FetchResult:
        fetch_cfg = self.settings.enrich.fetch
        http_cfg = self.settings.http
        host = urlparse(url).netloc.lower()
        backend = str(fetch_cfg.backend or "auto").lower()

        cache_allowed = not _should_disable_cache(
            fetch_cfg=fetch_cfg, http_cfg=http_cfg
        )
        cache_key = _hash_key(
            {
                "url": url,
                "kind": "fetch",
                "strategy": backend,
            }
        )

        with self.span("fetch.auto", url=url, strategy=backend) as sp:
            if cache_allowed:
                cached = await self._cache.aget(namespace="fetch", key=cache_key)
                if cached:
                    sp.set_attr("cache_hit", True)
                    payload = _decode_fetch_cache(cached)
                    return FetchResult(
                        url=str(payload.get("url") or url),
                        status_code=int(payload["status_code"]),
                        content_type=payload.get("content_type"),
                        content=bytes.fromhex(payload["content_hex"]),
                    )
            sp.set_attr("cache_hit", False)

            await self._rl.acquire(host=host)
            try:
                attempt = await self._fetch_useful(url=url, strategy=backend, span=sp)
            finally:
                await self._rl.release(host=host)

            blocked = self._is_blocked(attempt)
            if cache_allowed and not blocked:
                payload = {
                    "status_code": int(attempt.status_code),
                    "content_type": attempt.content_type,
                    "content_hex": attempt.content.hex(),
                    "url": attempt.url,
                    "strategy_used": attempt.strategy_used,
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
            )

    async def _fetch_useful(
        self, *, url: str, strategy: str, span: SpanBase
    ) -> FetchAttempt:
        if strategy == "httpx":
            res = await self._httpx.fetch_attempt(url=url, profile="browser", span=span)
            if await self._is_useful(res):
                return res
            raise RuntimeError("fetch_unusable:httpx")
        if strategy == "curl_cffi":
            if self._curl is None:
                raise RuntimeError(
                    "curl_cffi fetch strategy requested but curl fetcher not available"
                )
            res = await self._curl.fetch_attempt(url=url, span=span)
            if await self._is_useful(res):
                return res
            raise RuntimeError("fetch_unusable:curl_cffi")

        last: FetchAttempt | None = None
        last_useful = False

        for prof in ("browser", "compat"):
            res = await self._httpx.fetch_attempt(url=url, profile=prof, span=span)
            last = res
            last_useful = await self._is_useful(res)
            if last_useful:
                return res

        if self._curl is not None:
            res = await self._curl.fetch_attempt(url=url, span=span)
            last = res
            last_useful = await self._is_useful(res)
            if last_useful:
                return res

        assert last is not None
        if last_useful:
            return last
        raise RuntimeError("fetch_unusable:auto")

    def _is_blocked(self, res: FetchAttempt) -> bool:
        try:
            s = res.content[:_BLOCKED_SCAN_BYTES].decode("utf-8", errors="ignore")
        except Exception:
            return False
        return bool(_BLOCKED_RE.search(s))

    async def _is_useful(self, res: FetchAttempt) -> bool:
        status = int(res.status_code or 0)
        if status < 200 or status >= 400:
            return False
        if self._is_blocked(res):
            return False
        return len(res.content or b"") >= _MIN_USEFUL_BYTES


__all__ = ["AutoFetcher"]
