from __future__ import annotations

import json
from typing import Any, Literal

import anyio

from src.serpsage.core import AsyncSearcher, SearchConfig, Searcher


def main(
    query: str,
    depth: Literal["simple", "low", "medium", "high"] = "low",
    max_results: int = 1,
    type: Literal["json", "markdown"] = "json",
) -> dict[str, Any]:
    """Convenience entry point for simple usage."""

    cfg = SearchConfig.load("search_config_example.yaml")
    searcher = Searcher(cfg)

    if type == "markdown":
        result = searcher.search_markdown(query, depth, max_results=max_results)
    else:
        result = json.dumps(
            searcher.search_json(
                query,
                depth,  # depth: simple|low|medium|high
                max_results=max_results,
                chunk_target_chars=1200,
                chunk_overlap_sentences=1,
            ),
            indent=2,
            ensure_ascii=False,
        )
    return {"search_result": result}


async def async_main(
    query: str,
    depth: Literal["simple", "low", "medium", "high"] = "low",
    max_results: int = 1,
    type: Literal["json", "markdown"] = "json",
) -> dict[str, Any]:
    """Convenience entry point for simple usage."""

    cfg = SearchConfig.load("search_config_example.yaml")
    async with AsyncSearcher(cfg) as searcher:
        if type == "markdown":
            result = await searcher.asearch_markdown(
                query, depth, max_results=max_results
            )
        else:
            result = json.dumps(
                await searcher.asearch_json(
                    query,
                    depth,  # depth: simple|low|medium|high
                    max_results=max_results,
                    chunk_target_chars=1200,
                    chunk_overlap_sentences=1,
                ),
                indent=2,
                ensure_ascii=False,
            )
    return {"search_result": result}


if __name__ == "__main__":
    print(
        anyio.run(async_main, "初音ミク 新曲 2025", "low", 5, "markdown")[
            "search_result"
        ]
    )
