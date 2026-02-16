from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from bs4.element import Tag

_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "button",
    "input",
    "select",
    "textarea",
    "form",
    "meta",
}
_HARD_NOISE_TAGS = {"nav", "header", "footer"}
_HARD_NOISE_ROLE = {"navigation", "banner", "contentinfo", "search"}
_NOISE_PATTERNS = re.compile(
    r"(cookie|consent|subscribe|signin|login|toolbar|promo|banner|share|social|"
    r"advert|sponsor|popup|modal|dialog|pager|tracking|breadcrumb|newsletter|"
    r"related|recommend|citation|bibliograph|extra-services|labstabs|toggle|"
    r"search|searchbox|site-search|autocomplete|ads?|ad-|ad_|sponsored|"
    r"promotion|promoted|marketing|outbrain|taboola)",
    re.IGNORECASE,
)
_SECONDARY_PATTERNS = re.compile(
    r"(sidebar|related|recommend|comment|discussion|thread|reply|answer|faq|"
    r"supplement|appendix|toc|table-of-contents|index|more-like-this)",
    re.IGNORECASE,
)
_PRIMARY_HINT_PATTERNS = re.compile(
    r"(content|article|post|story|doc|documentation|readme|main|body|entry|answer)",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[,.!?;:\u3002\uff01\uff1f\uff1b]")
_SEARCH_HINT_RE = re.compile(
    r"(search|searchbox|site-search|autocomplete|query|type to search)",
    re.IGNORECASE,
)
_AD_HINT_RE = re.compile(
    r"(ads?|ad-|ad_|sponsored|sponsor|promotion|promoted|outbrain|taboola)",
    re.IGNORECASE,
)
_NOISE_DATA_ATTR_KEYS = (
    "data-testid",
    "data-test",
    "data-track",
    "data-component",
    "data-slot",
    "data-module",
    "data-ad",
    "data-ads",
    "data-ad-slot",
    "data-ad-client",
    "data-google-query-id",
)


def parse_html_document(html_doc: str) -> BeautifulSoup:
    return BeautifulSoup(html_doc, "html.parser")


def cleanup_dom(
    soup: BeautifulSoup,
    *,
    keep_semantic_tags: set[str] | None = None,
) -> None:
    keep_semantic = {str(tag).strip().lower() for tag in (keep_semantic_tags or set())}
    for tag in list(soup.find_all(_DROP_TAGS)):
        with contextlib.suppress(Exception):
            tag.decompose()
    for tag in list(soup.find_all(True)):
        with contextlib.suppress(Exception):
            name = (tag.name or "").lower()
            role = str(tag.get("role") or "").lower()
            name_sem = _semantic_tag_for_name(name)
            role_sem = _semantic_tag_for_role(role)
            sem = name_sem or role_sem
            if name in _HARD_NOISE_TAGS and sem not in keep_semantic:
                tag.decompose()
                continue
            if role in _HARD_NOISE_ROLE and sem not in keep_semantic:
                tag.decompose()
                continue

            ident = " ".join(
                [
                    str(tag.get("id") or ""),
                    " ".join(tag.get("class") or []),
                    str(tag.get("aria-label") or ""),
                ]
            ).strip()
            if ident and _NOISE_PATTERNS.search(ident):
                ident_sem = sem
                if ident_sem is None and "banner" in ident.lower():
                    ident_sem = "banner"
                if ident_sem not in keep_semantic:
                    tag.decompose()
                    continue

            if sem not in keep_semantic and _looks_like_search_or_ad(tag):
                tag.decompose()
                continue

            hidden = str(tag.get("aria-hidden") or "").lower() == "true"
            style = str(tag.get("style") or "").lower()
            if hidden or "display:none" in style or "visibility:hidden" in style:
                tag.decompose()
                continue

            if sem not in keep_semantic and _looks_like_link_farm(tag):
                tag.decompose()


def text_len(tag: Tag | BeautifulSoup) -> int:
    return len(clean_whitespace(tag.get_text(" ", strip=True)))


def is_secondary_container(tag: Tag) -> bool:
    name = (tag.name or "").lower()
    if name == "aside":
        return True
    role = str(tag.get("role") or "").lower()
    if role == "complementary":
        return True
    ident = " ".join(
        [str(tag.get("id") or ""), " ".join(tag.get("class") or [])]
    )
    return bool(ident and _SECONDARY_PATTERNS.search(ident))


def is_noise_container(tag: Tag) -> bool:
    name = (tag.name or "").lower()
    if name in _HARD_NOISE_TAGS:
        return True
    role = str(tag.get("role") or "").lower()
    if role in _HARD_NOISE_ROLE:
        return True
    ident = " ".join(
        [str(tag.get("id") or ""), " ".join(tag.get("class") or [])]
    )
    if ident and _NOISE_PATTERNS.search(ident):
        return True
    return _looks_like_search_or_ad(tag)


def is_descendant_of(tag: Tag, ancestor: Tag | BeautifulSoup) -> bool:
    parent = tag.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def score_primary_candidate(tag: Tag) -> float:
    text = clean_whitespace(tag.get_text(" ", strip=True))
    chars = len(text)
    if chars < 140:
        return -1.0
    links = " ".join(a.get_text(" ", strip=True) for a in tag.find_all("a"))
    link_density = len(links) / max(1, chars)
    punct_density = len(_PUNCT_RE.findall(text)) / max(1, chars)
    heading_count = len(tag.find_all(["h1", "h2", "h3"]))
    p_count = len(tag.find_all("p"))
    ident = " ".join(
        [str(tag.get("id") or ""), " ".join(tag.get("class") or [])]
    )
    hint_bonus = 1.0 if (ident and _PRIMARY_HINT_PATTERNS.search(ident)) else 0.0
    noise_penalty = 1.0 if (ident and _NOISE_PATTERNS.search(ident)) else 0.0
    return (
        chars * (1.0 - min(0.94, link_density))
        + heading_count * 52
        + p_count * 11
        + punct_density * 900
        + hint_bonus * 180
        - noise_penalty * 240
    )


def _semantic_tag_for_name(name: str) -> str | None:
    if name == "header":
        return "header"
    if name == "nav":
        return "navigation"
    if name == "footer":
        return "footer"
    return None


def _semantic_tag_for_role(role: str) -> str | None:
    if role == "navigation":
        return "navigation"
    if role == "banner":
        return "banner"
    if role == "contentinfo":
        return "footer"
    return None


def _looks_like_link_farm(tag: Tag) -> bool:
    name = (tag.name or "").lower()
    if name not in {"div", "section", "aside", "ul", "ol", "nav"}:
        return False

    text = clean_whitespace(tag.get_text(" ", strip=True))
    chars = len(text)
    if chars < 140:
        return False

    links = tag.find_all("a")
    link_count = len(links)
    if link_count < 8:
        return False

    link_text = clean_whitespace(
        " ".join(a.get_text(" ", strip=True) for a in links)
    )
    link_density = len(link_text) / max(1, chars)
    punct_density = len(_PUNCT_RE.findall(text)) / max(1, chars)
    paragraph_count = len(tag.find_all("p"))
    has_structure = bool(tag.find(["table", "pre", "code"]))

    return (
        not has_structure
        and link_density >= 0.38
        and punct_density <= 0.018
        and paragraph_count <= 2
    )


def _looks_like_search_or_ad(tag: Tag) -> bool:
    role = str(tag.get("role") or "").lower()
    if role == "search":
        return True

    type_attr = str(tag.get("type") or "").lower()
    if type_attr == "search":
        return True

    aria_label = str(tag.get("aria-label") or "")
    placeholder = str(tag.get("placeholder") or "")
    ident = " ".join(
        [
            str(tag.get("id") or ""),
            " ".join(tag.get("class") or []),
            aria_label,
            placeholder,
            str(tag.get("name") or ""),
        ]
    ).strip()
    if ident and (_SEARCH_HINT_RE.search(ident) or _AD_HINT_RE.search(ident)):
        return True

    for key in _NOISE_DATA_ATTR_KEYS:
        value = str(tag.get(key) or "").strip()
        if value and (_SEARCH_HINT_RE.search(value) or _AD_HINT_RE.search(value)):
            return True
    return False
