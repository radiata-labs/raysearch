from __future__ import annotations

from serpsage.settings.models import FetchSettings


def browser_headers(
    fetch_cfg: FetchSettings, *, profile: str | None = None
) -> dict[str, str]:
    ua = str(fetch_cfg.user_agent)

    headers: dict[str, str] = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
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

    extra = fetch_cfg.extra_headers or {}
    for k, v in extra.items():
        if not k:
            continue
        headers[str(k)] = str(v)

    return headers


def get_delay_s(base_ms: int) -> float:
    return min(base_ms, 100) / 1000.0


def parse_retry_after_s(v: str | None) -> float | None:
    if not v:
        return None
    v = v.strip()
    if not v:
        return None
    if v.isdigit():
        return float(int(v))
    return None


__all__ = ["get_delay_s", "parse_retry_after_s"]
