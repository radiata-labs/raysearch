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


__all__ = ["browser_headers"]
