"""Blend provider that fans out a query across multiple provider routes.

Unlike the other modules in this package, Blend is a local meta-provider rather
than an upstream search engine adapter:

- it discovers enabled provider routes from dependency injection
- it can include or exclude routes statically or per request
- it merges duplicate URLs across providers
- it reranks the merged result set through the configured ranker

Configuration
=============

Example configuration in this project:

.. code:: yaml

   blend:
     enabled: true
     default_sources: ["google", "duckduckgo"]
     include_sources: ["google", "duckduckgo", "wikipedia"]
     exclude_sources: []

Notes
=====

- Blend does not talk to an external endpoint directly.
- It exists to combine provider strengths while preserving the shared
  ``SearchProviderBase`` interface.
- The final ordering is determined by local ranking, not provider order alone.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import override

import anyio
from pydantic import Field, field_validator

from serpsage.components.loads import ComponentRegistry
from serpsage.components.provider.base import (
    ProviderConfigBase,
    ProviderMeta,
    SearchProviderBase,
)
from serpsage.components.rank.base import RankerBase
from serpsage.dependencies import CACHE_TOKEN, Depends, solve_dependencies
from serpsage.models.components.provider import SearchProviderResult
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import canonicalize_url, clean_whitespace

if TYPE_CHECKING:
    from serpsage.settings.models import AppSettings


async def provider_routes_factory(
    cache: dict[Any, Any] = Depends(CACHE_TOKEN),
    registry: ComponentRegistry = Depends(),
) -> tuple[SearchProviderBase[ProviderConfigBase], ...]:
    """Factory function: collect all enabled provider routes (excluding blend)."""
    routes: list[SearchProviderBase[ProviderConfigBase]] = []
    for spec in registry.enabled_specs("provider"):
        if spec.name == "blend":
            continue
        if not issubclass(spec.cls, SearchProviderBase):
            continue
        instance = await solve_dependencies(spec.cls, dependency_cache=cache)
        if isinstance(instance, SearchProviderBase):
            routes.append(instance)
    return tuple(routes)


def _normalize_source_names(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, Iterable):
        raw_items = list(value)
    else:
        raise TypeError("blend sources must be an iterable of strings")
    return {
        token for item in raw_items if (token := clean_whitespace(str(item)).casefold())
    }


class BlendProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "blend"

    default_sources: set[str] = Field(default_factory=set)
    include_sources: set[str] = Field(default_factory=set)
    exclude_sources: set[str] = Field(default_factory=set)
    original_rank_weight: float = 0.5  # Weight for original provider ranking (0-1)

    @field_validator(
        "default_sources", "include_sources", "exclude_sources", mode="before"
    )
    @classmethod
    def _normalize_sources(cls, value: object) -> set[str]:
        return _normalize_source_names(value)

    @field_validator("original_rank_weight", mode="before")
    @classmethod
    def _validate_original_rank_weight(cls, value: object) -> float:
        if value is None:
            return 0.5
        try:
            v = float(str(value))
        except (TypeError, ValueError):
            return 0.5
        if v < 0 or v > 1:
            raise ValueError("original_rank_weight must be between 0 and 1")
        return v


class BlendProvider(SearchProviderBase[BlendProviderConfig]):
    def __init__(
        self,
        *,
        routes: tuple[SearchProviderBase[ProviderConfigBase], ...] = Depends(
            provider_routes_factory
        ),
        ranker: RankerBase = Depends(),
    ) -> None:
        self.ranker = ranker
        self._all_routes: dict[str, SearchProviderBase[ProviderConfigBase]] = {}
        for route in routes:
            name = route.config.name
            if name == "blend":
                continue
            if not isinstance(getattr(route, "meta", None), ProviderMeta):
                continue
            if name in self._all_routes:
                raise ValueError(f"duplicate provider route `{name}`")
            self._all_routes[name] = route
        self._routes = self._configured_routes()

    def get_routes(self) -> tuple[SearchProviderBase[ProviderConfigBase], ...]:
        return self._routes

    @override
    async def _asearch(
        self,
        *,
        query: str,
        limit: int | None = None,
        language: str = "",
        location: str = "",
        moderation: bool = True,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        **kwargs: Any,
    ) -> list[SearchProviderResult]:
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")
        runtime_include = _normalize_source_names(kwargs.get("include_sources"))
        runtime_exclude = _normalize_source_names(kwargs.get("exclude_sources"))
        runtime_include_provided = bool(runtime_include)
        runtime_exclude_provided = bool(runtime_exclude)
        effective_include = runtime_include
        if (
            not runtime_include_provided
            and not runtime_exclude_provided
            and self.config.default_sources
        ):
            effective_include = set(self.config.default_sources)
        selected = self._selected_routes(
            include=effective_include,
            exclude=runtime_exclude,
        )
        if not selected:
            return []
        provider_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"default_sources", "include_sources", "exclude_sources"}
        }

        outputs: list[list[SearchProviderResult] | None] = [None] * len(selected)

        async def run_one(
            index: int, provider: SearchProviderBase[ProviderConfigBase]
        ) -> None:
            try:
                outputs[index] = await provider._asearch(
                    query=normalized_query,
                    limit=limit,
                    language=language,
                    location=location,
                    moderation=moderation,
                    start_published_date=start_published_date,
                    end_published_date=end_published_date,
                    **provider_kwargs,
                )
            except Exception:
                outputs[index] = None

        async with anyio.create_task_group() as tg:
            for index, provider in enumerate(selected):
                tg.start_soon(run_one, index, provider)

        merged: list[SearchProviderResult] = []
        positions: list[int] = []  # Track positions separately
        seen: set[str] = set()
        for items in outputs:
            for idx, item in enumerate(items or ()):
                key = canonicalize_url(item.url) or item.url.casefold()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
                positions.append(idx + 1)  # 1-indexed position from provider
        if not merged:
            return []

        texts = [f"{item.title} {item.title} {item.snippet}".strip() for item in merged]
        scores = await self.ranker.score_texts(
            texts,
            query=normalized_query,
            query_tokens=tokenize_for_query(normalized_query),
            mode="retrieve",
        )

        # Blend with original rank if weight > 0
        original_rank_weight = float(self.config.original_rank_weight)
        if original_rank_weight > 0:
            # Calculate position scores (reciprocal rank)
            position_scores = [
                1.0 / max(1, pos) if pos > 0 else 0.5 for pos in positions
            ]
            # Normalize scores
            max_score = max(scores) if scores and max(scores) > 0 else 1.0
            max_pos = (
                max(position_scores)
                if position_scores and max(position_scores) > 0
                else 1.0
            )
            norm_scores = [s / max_score for s in scores]
            norm_pos = [p / max_pos for p in position_scores]
            # Blend: weight for original_rank, (1 - weight) for semantic
            semantic_weight = 1.0 - original_rank_weight
            scores = [
                semantic_weight * ns + original_rank_weight * np
                for ns, np in zip(norm_scores, norm_pos, strict=False)
            ]

        ranked = sorted(
            enumerate(merged),
            key=lambda pair: (
                -(float(scores[pair[0]]) if pair[0] < len(scores) else 0.0),
                pair[0],
            ),
        )
        results = [item for _, item in ranked]
        if limit is None:
            return results
        return results[: max(1, int(limit))]

    def _selected_routes(
        self,
        *,
        include: set[str],
        exclude: set[str],
    ) -> list[SearchProviderBase[ProviderConfigBase]]:
        selected: list[SearchProviderBase[ProviderConfigBase]] = []
        for route in self._routes:
            name = route.config.name
            if include and name not in include:
                continue
            if name in exclude:
                continue
            selected.append(route)
        return selected

    def _configured_routes(self) -> tuple[SearchProviderBase[ProviderConfigBase], ...]:
        selected: list[SearchProviderBase[ProviderConfigBase]] = []
        for name, route in self._all_routes.items():
            if self.config.include_sources and name not in self.config.include_sources:
                continue
            if name in self.config.exclude_sources:
                continue
            selected.append(route)
        return tuple(selected)


def resolve_engine_selection_routes(
    *,
    settings: AppSettings,
    subsystem: Literal["search", "research", "answer"],
    provider: SearchProviderBase[ProviderConfigBase],
) -> tuple[ProviderMeta, ...]:
    subsystem_settings = getattr(settings, subsystem, None)
    if not bool(getattr(subsystem_settings, "select_engines", False)):
        return ()
    if str(getattr(provider.config, "name", "") or "") != "blend":
        return ()
    raw_get_routes = getattr(provider, "get_routes", None)
    if not callable(raw_get_routes):
        return ()
    out: list[ProviderMeta] = []
    for route in cast(
        "tuple[SearchProviderBase[ProviderConfigBase], ...]", raw_get_routes()
    ):
        meta = getattr(route, "meta", None)
        if isinstance(meta, ProviderMeta):
            out.append(meta)
    return tuple(out)


def build_engine_selection_context(*, routes: tuple[ProviderMeta, ...]) -> str:
    if not routes:
        return ""
    return "\n".join(
        [
            "ENGINE_SELECTION_BASE_CONTEXT:",
            "- You are choosing include_sources for BlendProvider on a per-query basis.",
            "- include_sources is a set-like list of engine names. Use names exactly as listed.",
            "- include_sources=[] is the preferred default for generic discovery because it keeps Blend default all-open routing.",
            "- Restrict include_sources only when the query clearly benefits from targeted evidence routes.",
            "- If the query needs general web coverage, prefer multiple general engines or [].",
            "- Choose engines using both description and preference: description tells you what the engine covers, preference tells you what query shape it handles best.",
            "- Aim for the smallest engine set that effectively addresses the query, but consider including multiple engines if they provide complementary coverage or diverse perspectives that together enhance the answer.",
            "- Avoid adding engines with largely redundant coverage, unless the query explicitly benefits from broader recall (e.g., comprehensive search).",
            "- For queries that require diverse sources (e.g., news from different outlets, multiple viewpoints, or cross-validation), selecting several relevant engines is recommended.",
            "- If you are unsure which engine subset is best, keep include_sources empty.",
            "ENGINE_CATALOG:",
            *[
                f"- {meta.name}: description={meta.description}; preference={meta.preference}; "
                f"categories={', '.join(meta.categories)}; website={meta.website}"
                for meta in routes
            ],
        ]
    )


__all__ = [
    "BlendProvider",
    "BlendProviderConfig",
    "build_engine_selection_context",
    "provider_routes_factory",
    "resolve_engine_selection_routes",
]
