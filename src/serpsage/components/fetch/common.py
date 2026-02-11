from __future__ import annotations

from typing import Any


def parse_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return (content_type.split(";", 1)[0] or "").strip().lower()


def looks_like_html(sample: bytes) -> bool:
    try:
        from serpsage.components.extract.utils import (
            looks_like_html as _looks_like_html,  # noqa: PLC0415
        )

        return bool(_looks_like_html(sample))
    except Exception:
        head = (sample or b"")[:8192].lower()
        return any(
            tok in head for tok in (b"<!doctype", b"<html", b"<body", b"</p", b"</div")
        )


def browser_headers(fetch_cfg: Any, *, profile: str | None = None) -> dict[str, str]:
    ua = str(fetch_cfg.user_agent)
    lang = str(getattr(fetch_cfg, "accept_language", "") or "en")
    enc = (
        "gzip, deflate"
        if bool(getattr(fetch_cfg, "disable_br", True))
        else "gzip, deflate, br"
    )

    headers: dict[str, str] = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": lang,
        "Accept-Encoding": enc,
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    }

    if profile == "browser":
        headers.update(
            {
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            }
        )

    extra = getattr(fetch_cfg, "extra_headers", None) or {}
    for k, v in extra.items():
        if not k:
            continue
        headers[str(k)] = str(v)

    return headers


def truncate_bytes(data: bytes, *, max_bytes: int) -> tuple[bytes, bool]:
    if max_bytes <= 0:
        return data, False
    if len(data) <= max_bytes:
        return data, False
    return data[:max_bytes], True


__all__ = ["browser_headers", "looks_like_html", "parse_content_type", "truncate_bytes"]
