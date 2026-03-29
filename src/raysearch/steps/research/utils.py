"""Research step utility functions.

This module contains pure utility functions used across research steps.
It is one of the two allowed pure-function files (along with prompt.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from raysearch.models.steps.research import (
    ResearchSource,
)
from raysearch.settings.models import AppSettings
from raysearch.tokenize import tokenize_for_query
from raysearch.utils import clean_whitespace

if TYPE_CHECKING:
    from raysearch.components.rank.base import RankerBase
    from raysearch.models.steps.research import (
        ResearchStepContext,
        RoundStepContext,
    )


_TRACKING_QUERY_KEYS = {"gclid", "fbclid", "msclkid"}
_TRACKING_QUERY_PREFIXES = ("utm_",)

# Authority scoring hints
_LOW_AUTHORITY_HOST_HINTS = (
    "medium.com",
    "substack.com",
    "blogspot.com",
    "wordpress.com",
)
_HIGH_AUTHORITY_HOST_HINTS = (
    "github.com",
    "gitlab.com",
    "huggingface.co",
    "arxiv.org",
    "doi.org",
    "w3.org",
    "ietf.org",
    "iso.org",
)
_HIGH_AUTHORITY_PATH_HINTS = (
    "/docs",
    "/documentation",
    "/api",
    "/reference",
    "/spec",
    "/specification",
    "/manual",
    "/guide",
    "/papers",
    "/repository",
)
_HIGH_AUTHORITY_TITLE_HINTS = (
    "official",
    "documentation",
    "reference",
    "api",
    "spec",
    "specification",
    "standard",
    "repository",
    "paper",
    "preprint",
    "whitepaper",
)

# Reranking constants
_MAX_CONTENT_CHARS = 4000
_MAX_OVERVIEW_CHARS = 1800


def resolve_research_model(
    *,
    settings: AppSettings,
    stage: str,
    fallback: str,
) -> str:
    """Resolve the appropriate research model for a given stage."""
    model_settings = settings.research.models
    stage_to_model = {
        "plan": model_settings.plan,
        "link_select": model_settings.link_select,
        "overview": model_settings.abstract_analyze,
        "abstract": model_settings.abstract_analyze,
        "content": model_settings.content_analyze,
        "synthesize": model_settings.synthesize,
        "markdown": model_settings.markdown,
    }
    configured_model = clean_whitespace(stage_to_model.get(stage, ""))
    return configured_model or fallback


def canonicalize_url(raw_url: str) -> str:
    """Normalize a URL by removing tracking parameters and standardizing format."""
    token = clean_whitespace(raw_url)
    if not token:
        return ""
    try:
        parsed = urlsplit(token)
    except Exception:  # noqa: S112
        return token
    scheme = clean_whitespace(parsed.scheme).lower() or "https"
    host = clean_whitespace(parsed.netloc).lower()
    path = str(parsed.path or "/").strip() or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/":
        path = path.rstrip("/") or "/"
    pairs: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        norm_key = clean_whitespace(key)
        if not norm_key:
            continue
        key_lc = norm_key.casefold()
        if key_lc in _TRACKING_QUERY_KEYS:
            continue
        if any(key_lc.startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES):
            continue
        pairs.append((norm_key, clean_whitespace(value)))
    pairs.sort(key=lambda item: (item[0].casefold(), item[1]))
    query = urlencode(pairs, doseq=True)
    return urlunsplit((scheme, host, path, query, ""))


def source_authority_score(source: ResearchSource) -> float:
    """Compute authority score for a source based on URL patterns."""
    url = clean_whitespace(source.canonical_url or source.url).casefold()
    title = clean_whitespace(source.title).casefold()
    if not url:
        return 0.25
    try:
        parsed = urlsplit(url)
    except Exception:  # noqa: S112
        return 0.25
    host = clean_whitespace(parsed.netloc).casefold()
    path = clean_whitespace(parsed.path).casefold()
    score = 0.35
    if host.endswith((".gov", ".edu")):
        score = max(score, 0.95)
    if host.startswith(("docs.", "developer.")):
        score = max(score, 0.88)
    if any(token in host for token in _HIGH_AUTHORITY_HOST_HINTS):
        score = max(score, 0.90)
    if any(
        path.startswith(prefix) or f"{prefix}/" in path
        for prefix in _HIGH_AUTHORITY_PATH_HINTS
    ):
        score = max(score, 0.82)
    if any(token in title for token in _HIGH_AUTHORITY_TITLE_HINTS):
        score = max(score, 0.78)
    if any(token in host for token in _LOW_AUTHORITY_HOST_HINTS):
        score = min(score, 0.25)
    return min(1.0, max(0.05, score))


def pick_sources_by_ids(
    *,
    sources: list[ResearchSource],
    source_ids: list[int],
) -> list[ResearchSource]:
    """Pick sources by their IDs."""
    source_by_id = {source.source_id: source for source in sources}
    out: list[ResearchSource] = []
    for source_id in source_ids:
        source = source_by_id.get(source_id)
        if source is None:
            continue
        out.append(source)
    return out


def _build_research_source_rank_text(source: ResearchSource) -> str:
    """Build text for ranking a research source."""
    parts = [
        f"title={clean_whitespace(source.title)}",
        f"url={clean_whitespace(source.url)}",
        f"overview={clean_whitespace(source.overview)[:_MAX_OVERVIEW_CHARS]}",
        f"content={clean_whitespace(source.content)[:_MAX_CONTENT_CHARS]}",
    ]
    return "\n".join(part for part in parts if part.strip())


async def rerank_research_sources(
    *,
    ctx: ResearchStepContext | RoundStepContext,
    ranker: RankerBase,
    sources: list[ResearchSource],
    query: str,
) -> list[ResearchSource]:
    """Rerank research sources by relevance."""
    if len(sources) <= 1:
        return list(sources)
    effective_query = clean_whitespace(query) or clean_whitespace(ctx.task.question)
    if not effective_query:
        return list(sources)
    texts = [_build_research_source_rank_text(source) for source in sources]
    try:
        scores = await ranker.score_texts(
            texts,
            query=effective_query,
            query_tokens=tokenize_for_query(effective_query),
            mode="rerank",
        )
    except Exception:
        return list(sources)
    ordered = sorted(
        enumerate(sources),
        key=lambda item: (
            -_score_at(scores=scores, idx=item[0]),
            -float(ctx.knowledge.source_scores.get(item[1].source_id, 0.0)),
            -float(source_authority_score(item[1])),
            int(item[1].source_id),
            item[0],
        ),
    )
    return [source for _, source in ordered]


def _score_at(*, scores: list[float], idx: int) -> float:
    return float(scores[idx]) if idx < len(scores) else 0.0


__all__ = [
    "canonicalize_url",
    "pick_sources_by_ids",
    "rerank_research_sources",
    "resolve_research_model",
    "source_authority_score",
]
