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

from serpsage.components.provider.base import (
    PROVIDER_ROUTES_TOKEN,
    ProviderConfigBase,
    ProviderMeta,
    SearchProviderBase,
)
from serpsage.components.rank.base import RankerBase
from serpsage.dependencies import Depends
from serpsage.models.components.provider import SearchProviderResult
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import canonicalize_url, clean_whitespace

if TYPE_CHECKING:
    from serpsage.settings.models import AppSettings


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

    include_sources: set[str] = Field(default_factory=set)
    exclude_sources: set[str] = Field(default_factory=set)

    @field_validator("include_sources", "exclude_sources", mode="before")
    @classmethod
    def _normalize_sources(cls, value: object) -> set[str]:
        return _normalize_source_names(value)


class BlendProvider(SearchProviderBase[BlendProviderConfig]):
    def __init__(
        self,
        *,
        routes: tuple[object, ...] = Depends(PROVIDER_ROUTES_TOKEN),
        ranker: RankerBase = Depends(),
    ) -> None:
        self.ranker = ranker
        self._all_routes: dict[str, SearchProviderBase[ProviderConfigBase]] = {}
        for route in cast("tuple[SearchProviderBase[ProviderConfigBase], ...]", routes):
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
        selected = self._selected_routes(
            include=runtime_include,
            exclude=runtime_exclude,
        )
        if not selected:
            return []
        provider_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"include_sources", "exclude_sources"}
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
        seen: set[str] = set()
        for items in outputs:
            for item in items or ():
                key = canonicalize_url(item.url) or item.url.casefold()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        if not merged:
            return []

        texts = [f"{item.title} {item.title} {item.snippet}".strip() for item in merged]
        scores = await self.ranker.score_texts(
            texts,
            query=normalized_query,
            query_tokens=tokenize_for_query(normalized_query),
            mode="retrieve",
        )
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
            "- Prefer the smallest engine set that still matches the evidence need; do not add engines with overlapping value unless broader recall is clearly useful.",
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
    "resolve_engine_selection_routes",
]
