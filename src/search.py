from __future__ import annotations

import json
from typing import Any, Literal

import anyio
from dotenv import load_dotenv

from serpsage import Engine, SearchRequest, load_settings

load_dotenv()


async def main(
    query: str,
    depth: Literal["simple", "low", "medium", "high"] = "low",
    max_results: int = 5,
) -> dict[str, Any]:
    settings = load_settings("src/search_config_example.yaml")
    req = SearchRequest(
        query=query,
        depth=depth,
        max_results=max_results,
        # overview=False,
    )

    async with Engine.from_settings(settings) as engine:
        resp = await engine.search(req)

    return {
        "search_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2),
    }


if __name__ == "__main__":
    import time

    t1 = time.time()
    out = anyio.run(main, "花谱寓話专辑有哪些歌", "high", 5)
    t2 = time.time()

    print(out["search_result"])
    print(f"Search took {t2 - t1:.2f} seconds")
