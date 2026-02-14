from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.extract.markdown.dom import (
    is_descendant_of,
    is_noise_container,
    is_secondary_container,
    score_primary_candidate,
    text_len,
)
from serpsage.components.extract.markdown.types import SectionBuckets

if TYPE_CHECKING:
    from bs4 import BeautifulSoup
    from bs4.element import Tag


def split_sections(
    *,
    soup: BeautifulSoup,
    min_primary_chars: int,
) -> SectionBuckets:
    primary = _pick_primary_root(soup=soup, min_primary_chars=min_primary_chars)
    secondary = _collect_secondary_roots(soup=soup, primary_root=primary)
    secondary = _dedupe_nested_roots(secondary)
    return SectionBuckets(primary_root=primary, secondary_roots=secondary)


def _pick_primary_root(*, soup: BeautifulSoup, min_primary_chars: int) -> Tag | BeautifulSoup:
    for sel in ("main", "article", '[role="main"]', "#content", "#main", ".content", ".article"):
        found = soup.select_one(sel)
        if found is not None and text_len(found) >= min_primary_chars:
            return found

    best: Tag | BeautifulSoup = soup
    best_score = -1.0
    for cand in soup.find_all(["article", "main", "section", "div"]):
        if is_noise_container(cand):
            continue
        score = score_primary_candidate(cand)
        if score > best_score:
            best_score = score
            best = cand
    return best


def _collect_secondary_roots(
    *,
    soup: BeautifulSoup,
    primary_root: Tag | BeautifulSoup,
) -> list[Tag]:
    out: list[Tag] = []
    for tag in soup.find_all(True):
        if tag is primary_root:
            continue
        if is_noise_container(tag):
            continue
        if not is_secondary_container(tag):
            continue
        if text_len(tag) < 20:
            continue
        # only capture nodes that are outside primary root; inside-primary secondary
        # blocks are already rendered by primary flow.
        if is_descendant_of(tag, primary_root):
            continue
        out.append(tag)
    return out


def _dedupe_nested_roots(roots: list[Tag]) -> list[Tag]:
    deduped: list[Tag] = []
    for node in roots:
        if any(is_descendant_of(node, keep) for keep in deduped):
            continue
        deduped.append(node)
    return deduped
