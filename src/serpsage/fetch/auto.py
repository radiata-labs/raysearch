from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import urlparse

from anyio import to_thread

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.protocols import Cache, Extractor, Fetcher, FetchResult
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.app.runtime import CoreRuntime
    from serpsage.fetch.curl_cffi import CurlCffiFetcher
    from serpsage.fetch.http import HttpxFetcher


@dataclass(frozen=True)
class SimpleFetchResult:
    url: str
    status_code: int
    content_type: str | None
    content: bytes


@dataclass
class AttemptResult:
    final_url: str
    status_code: int
    content_type: str | None
    content: bytes
    truncated: bool
    strategy_used: str
    content_encoding: str | None = None
    content_length_header: str | None = None


_BLOCKED_RE = re.compile(
    r"(cloudflare|cf-ray|cf-chl|just a moment|attention required|captcha|"
    r"verify you are human|access denied|please enable javascript|人机验证|访问受限)",
    re.IGNORECASE,
)


def _hash_key(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def _encode_fetch_cache(payload: dict[str, Any]) -> bytes:
    # Keep cache format JSON (compressed by SqliteCache); content is hex to keep it ascii-safe.
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _decode_fetch_cache(blob: bytes) -> dict[str, Any]:
    return json.loads(blob.decode("utf-8"))


def _parse_ct(content_type: str | None) -> str:
    if not content_type:
        return ""
    return (content_type.split(";", 1)[0] or "").strip().lower()


def _should_disable_cache(fetch_cfg: Any) -> bool:
    # If using proxy/cookies/auth-like headers, caching can leak private content across runs.
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


class AutoFetcher(WorkUnit, Fetcher):
    """Fetch URLs with an auto strategy geared towards "extractable blocks".

    - Use httpx first (browser-like headers), then fallback to curl_cffi if content is blocked/unusable.
    - Optionally validate by running extractor and requiring minimum blocks/text chars.
    - Cache only "useful" pages by default.
    """

    def __init__(
        self,
        *,
        rt: CoreRuntime,
        cache: Cache,
        rate_limiter: Any,  # RateLimiter, typed as Any to avoid import cycles in typing.
        httpx_fetcher: HttpxFetcher,
        curl_fetcher: CurlCffiFetcher | None,
        extractor: Extractor,
    ) -> None:
        super().__init__(rt=rt)
        self._cache = cache
        self._rl = rate_limiter
        self._httpx = httpx_fetcher
        self._curl = curl_fetcher
        self._extractor = extractor

    @override
    async def afetch(self, *, url: str) -> FetchResult:
        fetch_cfg = self.settings.enrich.fetch
        host = urlparse(url).netloc.lower()
        strategy = str(getattr(fetch_cfg, "strategy", "auto") or "auto")

        cache_allowed = not _should_disable_cache(fetch_cfg)
        cache_key = _hash_key(
            {
                "url": url,
                "kind": "fetch",
                "accept_language": str(getattr(fetch_cfg, "accept_language", "")),
                "strategy": strategy,
            }
        )

        with self.span("fetch.auto", url=url, strategy=strategy) as sp:
            if cache_allowed:
                cached = await self._cache.aget(namespace="fetch", key=cache_key)
                if cached:
                    sp.set_attr("cache_hit", True)
                    payload = _decode_fetch_cache(cached)
                    return SimpleFetchResult(
                        url=str(payload.get("final_url") or url),
                        status_code=int(payload["status_code"]),
                        content_type=payload.get("content_type"),
                        content=bytes.fromhex(payload["content_hex"]),
                    )
            sp.set_attr("cache_hit", False)

            await self._rl.acquire(host=host)
            try:
                attempt = await self._fetch_useful(url=url, strategy=strategy, span=sp)
            finally:
                await self._rl.release(host=host)

            # Cache only useful pages (and optionally blocked pages, if enabled).
            blocked = self._is_blocked(attempt)
            if cache_allowed and (not blocked or bool(fetch_cfg.cache_blocked_pages)):
                payload = {
                    "status_code": int(attempt.status_code),
                    "content_type": attempt.content_type,
                    "content_hex": attempt.content.hex(),
                    "final_url": attempt.final_url,
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

            return SimpleFetchResult(
                url=attempt.final_url,
                status_code=int(attempt.status_code),
                content_type=attempt.content_type,
                content=attempt.content,
            )

    async def _fetch_useful(
        self, *, url: str, strategy: str, span: Any
    ) -> AttemptResult:
        fetch_cfg = self.settings.enrich.fetch

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

        # auto: try httpx profiles first, then curl.
        max_total = max(1, int(fetch_cfg.max_attempts_total))
        attempts = 0

        last: AttemptResult | None = None
        last_useful = False

        # httpx attempts
        for prof in ("browser", "compat"):
            if attempts >= max_total:
                break
            attempts += 1
            res = await self._httpx.fetch_attempt(url=url, profile=prof, span=span)
            last = res
            last_useful = await self._is_useful(res)
            if last_useful:
                return res
            # If blocked/unusable, continue.

        # curl fallback
        if self._curl is not None and attempts < max_total:
            attempts += 1
            res = await self._curl.fetch_attempt(url=url, span=span)
            last = res
            last_useful = await self._is_useful(res)
            if last_useful:
                return res
            # Fall through to raise fetch_unusable:auto.

        # Nothing useful; return last httpx attempt (already has body if any).
        assert last is not None
        if last_useful:
            return last
        raise RuntimeError("fetch_unusable:auto")

    def _is_blocked(self, res: AttemptResult) -> bool:
        # Check a small prefix; blocked pages are typically HTML with key phrases.
        try:
            s = (
                res.content[
                    : max(4096, int(self.settings.enrich.fetch.sniff_html_bytes))
                ]
            ).decode("utf-8", errors="ignore")
        except Exception:
            return False
        return bool(_BLOCKED_RE.search(s))

    async def _is_useful(self, res: AttemptResult) -> bool:  # noqa: PLR0911
        fetch_cfg = self.settings.enrich.fetch

        ct = _parse_ct(res.content_type)
        allow = {
            str(x).strip().lower()
            for x in (fetch_cfg.allow_content_types or [])
            if str(x).strip()
        }
        if allow and ct and ct not in allow:  # noqa: SIM102
            # Allow mislabelled HTML if it looks like HTML.
            if not self._looks_like_html(
                res.content[: int(fetch_cfg.sniff_html_bytes)]
            ):
                return False

        if len(res.content or b"") < int(fetch_cfg.min_html_bytes):
            return False

        if self._is_blocked(res) and not bool(fetch_cfg.cache_blocked_pages):
            return False

        if not bool(fetch_cfg.validate_extractable):
            return True

        max_chars = int(fetch_cfg.validate_max_chars)
        sample = res.content if max_chars <= 0 else res.content[:max_chars]

        try:
            extracted = await to_thread.run_sync(
                lambda: self._extractor.extract(
                    url=res.final_url, content=sample, content_type=res.content_type
                )
            )
        except Exception:
            return False

        # Note: extracted.blocks is a property; any object should satisfy ExtractedText protocol.
        blocks = list(getattr(extracted, "blocks", []) or [])
        if len(blocks) < int(fetch_cfg.min_blocks):
            return False
        txt_chars = sum(len(str(b)) for b in blocks)
        return not txt_chars < int(fetch_cfg.min_text_chars)

    @staticmethod
    def _looks_like_html(sample: bytes) -> bool:
        try:
            from serpsage.extract.utils import looks_like_html  # noqa: PLC0415

            return bool(looks_like_html(sample))
        except Exception:
            head = (sample or b"")[:8192].lower()
            return any(
                tok in head
                for tok in (b"<!doctype", b"<html", b"<body", b"</p", b"</div")
            )


__all__ = ["AutoFetcher", "AttemptResult", "SimpleFetchResult"]
