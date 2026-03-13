from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast
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
            if not isinstance(getattr(type(route), "meta", None), ProviderMeta):
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
        locale: str = "",
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
            outputs[index] = await provider._asearch(
                query=normalized_query,
                limit=limit,
                locale=locale,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                **provider_kwargs,
            )

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


__all__ = ["BlendProvider", "BlendProviderConfig"]
