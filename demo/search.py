from __future__ import annotations

import json
import time
from typing import Any, Literal

import anyio
from dotenv import load_dotenv

from serpsage import (
    Engine,
    FetchAbstractsRequest,
    FetchOthersRequest,
    FetchOverviewRequest,
    FetchSubpagesRequest,
    SearchFetchRequest,
    SearchRequest,
)

load_dotenv()


async def main(
    query: str,
    mode: Literal["fast", "auto", "deep"] = "deep",
    max_results: int = 10,
) -> dict[str, Any]:
    req = SearchRequest(
        query=query,
        mode=mode,
        max_results=max_results,
        fetchs=SearchFetchRequest(
            abstracts=FetchAbstractsRequest(query=query, max_chars=400),
            subpages=FetchSubpagesRequest(max_subpages=None, subpage_keywords=query),
            overview=FetchOverviewRequest(query=query),
            others=FetchOthersRequest(max_links=5, max_image_links=5),
        ),
    )
    async with Engine.from_settings("demo/search_config_example.yaml") as engine:
        await anyio.sleep(1)
        t1 = time.time()
        resp = await engine.search(req)
        t2 = time.time()
        print(f"Search took {t2 - t1:.4f} seconds")
    return {
        "search_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2),
    }


if __name__ == "__main__":
    import time
    from pathlib import Path

    cache = Path("./.cache/.serpsage_events.jsonl")
    if cache.exists():
        cache.unlink()
    out = anyio.run(main, "latest llm papers", "auto", 5)
    print(out["search_result"])
