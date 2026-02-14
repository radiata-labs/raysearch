from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.extract.markdown.types import ExtractProfile

if TYPE_CHECKING:
    from serpsage.settings.models import AppSettings

_DEFAULT_ENGINE_ORDER = [
    "fastdom",
    "readability",
    "trafilatura",
    "justext",
    "boilerpy3",
]
_KNOWN_ENGINES = set(_DEFAULT_ENGINE_ORDER)


def build_extract_profile(*, settings: AppSettings) -> ExtractProfile:
    cfg = settings.fetch.extract
    enabled = {str(e).strip().lower() for e in (cfg.engines or []) if str(e).strip()}
    enabled = {x for x in enabled if x in _KNOWN_ENGINES}
    if not enabled:
        enabled = set(_DEFAULT_ENGINE_ORDER)

    raw_order = [str(x).strip().lower() for x in (cfg.engine_order or []) if str(x).strip()]
    order = [x for x in raw_order if x in _KNOWN_ENGINES]
    if not order:
        order = list(_DEFAULT_ENGINE_ORDER)
    if "fastdom" not in order:
        order = ["fastdom", *order]
    # engines acts as a deny/allow list
    order = [x for x in order if x in enabled or x == "fastdom"]

    max_markdown = int(max(8_000, cfg.max_markdown_chars))
    return ExtractProfile(
        enabled_engines=enabled,
        engine_order=order,
        engine_timeout_ms=max(350, int(cfg.engine_timeout_ms)),
        max_markdown_chars=max_markdown,
        max_html_chars=max_markdown * 3,
        min_plain_chars=max(120, int(cfg.min_plain_chars)),
        min_primary_chars=max(120, int(cfg.min_primary_chars)),
        min_total_chars_with_secondary=max(120, int(cfg.min_total_chars_with_secondary)),
        include_secondary_default=bool(cfg.include_secondary_content_default),
        collect_links_default=bool(cfg.collect_links_default),
        link_max_count=max(1, int(cfg.link_max_count)),
        link_keep_hash=bool(cfg.link_keep_hash),
    )
