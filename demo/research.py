"""Demo: ResearchRequest - comprehensive field showcase."""

from __future__ import annotations

import json
from typing import Any, Literal

import anyio
from dotenv import load_dotenv

from raysearch import Engine, ResearchRequest

load_dotenv()


async def main(
    themes: str,
    search_mode: Literal["research-fast", "research", "research-pro"] = "research",
) -> dict[str, Any]:
    """
    Demonstrate ResearchRequest.

    Args:
        themes: Research topic or question (required).
        search_mode: Research depth - research-fast/research/research-pro.

    Returns:
        Dict containing research result as JSON.
    """
    # ResearchRequest fields: themes (required), search_mode, json_schema
    req = ResearchRequest(
        themes=themes,
        search_mode=search_mode,
        json_schema=None,  # Optional: JSON Schema for structured output
    )
    async with Engine("demo/search_config_example.yaml") as engine:
        resp = await engine.research(req)
    return {
        "research_result": json.dumps(
            resp.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    }


if __name__ == "__main__":
    # Classic research question: Compare major programming languages
    out = anyio.run(
        main,
        "Compare Python, JavaScript, and Go for web backend development",
        "research-fast",
    )
    print(out["research_result"])
