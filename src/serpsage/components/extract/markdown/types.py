from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

from serpsage.models.extract import ExtractedLink

if TYPE_CHECKING:
    from bs4 import BeautifulSoup
    from bs4.element import Tag

StatValue: TypeAlias = int | float | str | bool
StatsMap: TypeAlias = dict[str, StatValue]
SectionName: TypeAlias = Literal["primary", "secondary"]


@dataclass(slots=True)
class SectionBuckets:
    primary_root: Tag | BeautifulSoup
    secondary_roots: list[Tag]


@dataclass(slots=True)
class CandidateDoc:
    markdown: str
    plain_text: str
    extractor_used: str
    quality_score: float
    warnings: list[str]
    stats: StatsMap
    primary_chars: int = 0
    secondary_chars: int = 0
    links: list[ExtractedLink] = field(default_factory=list)


@dataclass(slots=True)
class ExtractProfile:
    enabled_engines: set[str]
    engine_order: list[str]
    engine_timeout_ms: int
    max_markdown_chars: int
    max_html_chars: int
    min_plain_chars: int
    min_primary_chars: int
    min_total_chars_with_secondary: int
    include_secondary_default: bool
    collect_links_default: bool
    link_max_count: int
    link_keep_hash: bool
    quality_threshold: float = 0.48


__all__ = [
    "CandidateDoc",
    "ExtractProfile",
    "SectionBuckets",
    "SectionName",
    "StatsMap",
    "StatValue",
]
