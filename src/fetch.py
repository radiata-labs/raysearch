from __future__ import annotations

import json
from typing import Any

import anyio
from dotenv import load_dotenv

from serpsage import (
    Engine,
    FetchChunksRequest,
    FetchOverviewRequest,
    FetchRequest,
    load_settings,
)

load_dotenv()


async def main(
    url: str,
    query: str | None = None,
    overview: bool = False,
) -> dict[str, Any]:
    settings = load_settings("src/search_config_example.yaml")
    req = FetchRequest(
        url=url,
        content=True,
        chunks=FetchChunksRequest(query=query) if query else None,
        overview=FetchOverviewRequest(query=(query or url)) if overview else None,
    )
    async with Engine.from_settings(settings) as engine:
        resp = await engine.fetch(req)
    return {
        "fetch_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)
        + "\n\n"
        + resp.page.markdown,
    }


if __name__ == "__main__":
    out = anyio.run(main, "https://exa.ai/docs/reference/search-best-practices", None, False)
    print(out["fetch_result"])
