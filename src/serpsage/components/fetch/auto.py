from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import urlparse

from anyio import to_thread

from serpsage.components.fetch.common import looks_like_html, parse_content_type
from serpsage.contracts.services import CacheBase, ExtractorBase, FetcherBase
from serpsage.models.fetch import FetchAttempt, FetchResult
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.components.fetch.curl_cffi import CurlCffiFetcher
    from serpsage.components.fetch.http import HttpxFetcher


_BLOCKED_RE = re.compile(
    r"(cloudflare|cf-ray|cf-chl|just a moment|attention required|captcha|"
    r"verify you are human|access denied|please enable javascript|人机验证|访问受限)",
    re.IGNORECASE,
)


def _hash_key(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def _encode_fetch_cache(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _decode_fetch_cache(blob: bytes) -> dict[str, Any]:
    return json.loads(blob.decode("utf-8"))


def _should_disable_cache(fetch_cfg: Any) -> bool:
    if getattr(fetch_cfg, "proxy", None):
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
        rate_limiter: Any,
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
        common = fetch_cfg.common
        host = urlparse(url).netloc.lower()
        backend = str(fetch_cfg.backend or "auto").lower()

        cache_allowed = not _should_disable_cache(common)
        cache_key = _hash_key(
            {
                "url": url,
                "kind": "fetch",
                "accept_language": str(common.accept_language),
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
            if cache_allowed and (not blocked or bool(common.cache_blocked_pages)):
                payload = {
                    "status_code": int(attempt.status_code),
                    "content_type": attempt.content_type,
                    "content_hex": attempt.content.hex(),
                    "url": attempt.url,
                    "truncated": bool(attempt.truncated),
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
        self, *, url: str, strategy: str, span: Any
    ) -> FetchAttempt:
        fetch_cfg = self.settings.enrich.fetch
        auto_cfg = fetch_cfg.auto

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

        max_total = max(1, int(auto_cfg.max_attempts_total))
        attempts = 0

        last: FetchAttempt | None = None
        last_useful = False

        for prof in ("browser", "compat"):
            if attempts >= max_total:
                break
            attempts += 1
            res = await self._httpx.fetch_attempt(url=url, profile=prof, span=span)
            last = res
            last_useful = await self._is_useful(res)
            if last_useful:
                return res

        if self._curl is not None and attempts < max_total:
            attempts += 1
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
        common = self.settings.enrich.fetch.common
        try:
            s = res.content[: max(4096, int(common.sniff_html_bytes))].decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            return False
        return bool(_BLOCKED_RE.search(s))

    async def _is_useful(self, res: FetchAttempt) -> bool:  # noqa: PLR0911
        fetch_cfg = self.settings.enrich.fetch
        common = fetch_cfg.common

        ct = parse_content_type(res.content_type)
        allow = {
            str(x).strip().lower()
            for x in (common.allow_content_types or [])
            if str(x).strip()
        }
        if (
            allow
            and ct
            and ct not in allow
            and not looks_like_html(res.content[: int(common.sniff_html_bytes)])
        ):
            return False

        if len(res.content or b"") < int(common.min_html_bytes):
            return False

        if self._is_blocked(res) and not bool(common.cache_blocked_pages):
            return False

        if not bool(common.validate_extractable):
            return True

        max_chars = int(common.validate_max_chars)
        sample = res.content if max_chars <= 0 else res.content[:max_chars]

        try:
            extracted = await to_thread.run_sync(
                lambda: self._extractor.extract(
                    url=res.url, content=sample, content_type=res.content_type
                )
            )
        except Exception:
            return False

        blocks = list(getattr(extracted, "blocks", []) or [])
        if len(blocks) < int(common.min_blocks):
            return False
        txt_chars = sum(len(str(b)) for b in blocks)
        return not txt_chars < int(common.min_text_chars)


__all__ = ["AutoFetcher"]
