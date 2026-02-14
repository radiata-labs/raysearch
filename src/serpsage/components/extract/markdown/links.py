from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4.element import Tag

from serpsage.components.extract.markdown.dom import is_descendant_of
from serpsage.components.extract.markdown.types import SectionName
from serpsage.models.extract import ExtractedLink
from serpsage.text.normalize import clean_whitespace

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

    from serpsage.components.extract.markdown.types import SectionBuckets

_TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "msclkid",
}


def collect_links(
    *,
    soup: BeautifulSoup,
    base_url: str,
    buckets: SectionBuckets,
    include_secondary_content: bool,
    max_links: int,
    keep_hash: bool,
) -> list[ExtractedLink]:
    out: list[ExtractedLink] = []
    seen: set[tuple[str, str, SectionName]] = set()
    base_norm = _normalize_url(url=base_url, base_url=base_url, keep_hash=keep_hash)
    if base_norm is None:
        base_norm = base_url
    base_netloc = urlparse(base_norm).netloc.lower()

    position = 0
    for anchor in soup.find_all("a"):
        position += 1
        href = str(anchor.get("href") or "").strip()
        text = clean_whitespace(anchor.get_text(" ", strip=True))
        if not href or not text:
            continue

        normalized = _normalize_url(url=href, base_url=base_url, keep_hash=keep_hash)
        if not normalized:
            continue

        section = _section_for_tag(anchor, buckets=buckets)
        if section == "secondary" and not include_secondary_content:
            continue

        key = (normalized, text.lower(), section)
        if key in seen:
            continue
        seen.add(key)

        parsed = urlparse(normalized)
        rel = {str(x).lower() for x in (anchor.get("rel") or [])}
        same_page = _strip_fragment(normalized) == _strip_fragment(base_norm)
        out.append(
            ExtractedLink(
                url=normalized,
                anchor_text=text,
                section=section,
                is_internal=(parsed.netloc.lower() == base_netloc) if parsed.netloc else True,
                nofollow=("nofollow" in rel),
                same_page=same_page,
                source_hint=_source_hint(anchor),
                position=position,
            )
        )
        if len(out) >= max(1, int(max_links)):
            break

    return out


def _normalize_url(*, url: str, base_url: str, keep_hash: bool) -> str | None:
    raw = (url or "").strip()
    if not raw:
        return None
    if raw.lower().startswith(("javascript:", "mailto:", "tel:", "data:")):
        return None

    try:
        joined = str(urljoin(base_url, raw))
        parsed = urlparse(joined)
        if parsed.scheme not in {"http", "https"}:
            return None

        clean_pairs = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_KEYS and not k.lower().startswith("utm_")
        ]
        fragment = parsed.fragment if keep_hash else ""
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                urlencode(clean_pairs, doseq=True),
                fragment,
            )
        )
    except Exception:
        return None


def _strip_fragment(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )


def _section_for_tag(tag: Tag, *, buckets: SectionBuckets) -> SectionName:
    for sec in buckets.secondary_roots:
        if tag is sec or is_descendant_of(tag, sec):
            return "secondary"
    if tag is buckets.primary_root or is_descendant_of(tag, buckets.primary_root):
        return "primary"
    return "secondary"


def _source_hint(tag: Tag) -> str:
    cur: Tag | None = tag
    hops = 0
    while cur is not None and hops < 4:
        ident = " ".join([str(cur.get("id") or ""), " ".join(cur.get("class") or [])]).strip()
        if ident:
            return ident[:120]

        parent = cur.parent
        if not isinstance(parent, Tag):
            break
        cur = parent
        hops += 1
    return "unknown"
