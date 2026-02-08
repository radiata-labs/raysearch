from __future__ import annotations

import json
from typing import Any

import anyio

from search_core import AsyncSearcher, SearchConfig, Searcher


def main(query: str, max_results: int = 1) -> dict[str, Any]:
    """Convenience entry point for simple usage."""

    cfg = SearchConfig.load()
    pipeline = Searcher(cfg)
    result = json.dumps(
        pipeline.search_json(
            query,
            "high",  # depth: simple|low|medium|high
            max_results=max_results,
            chunk_target_chars=1200,
            chunk_overlap_sentences=1,
        ),
        indent=2,
        ensure_ascii=False,
    )
    return {"search_result": result}


async def async_main(query: str, max_results: int = 1) -> dict[str, Any]:
    """Convenience entry point for simple usage."""

    cfg = SearchConfig.load()
    async with AsyncSearcher(cfg) as pipeline:
        result = json.dumps(
            await pipeline.asearch_json(
                query,
                "high",  # depth: simple|low|medium|high
                max_results=max_results,
                chunk_target_chars=1200,
                chunk_overlap_sentences=1,
            ),
            indent=2,
            ensure_ascii=False,
        )
    return {"search_result": result}


if __name__ == "__main__":
    print(main("初音ミク 新曲 2025", 5)["search_result"])
    print(anyio.run(async_main, "初音ミク 新曲 2025", 5)["search_result"])
