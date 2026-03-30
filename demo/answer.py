"""Demo: AnswerRequest - comprehensive field showcase."""

from __future__ import annotations

import json
import time
from typing import Any

import anyio
from dotenv import load_dotenv

from raysearch import AnswerRequest, Engine

load_dotenv()


async def main(
    query: str,
    content: bool = False,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Demonstrate AnswerRequest.

    Args:
        query: The question to answer (required).
        content: Whether to fetch full content for deeper analysis.
        json_schema: Optional JSON Schema for structured output.

    Returns:
        Dict containing answer text and full JSON result.
    """
    # AnswerRequest fields: query (required), content, json_schema
    req = AnswerRequest(
        query=query,
        content=content,
        json_schema=json_schema,
    )
    async with Engine("demo/search_config_example.yaml") as engine:
        await anyio.sleep(1)
        t1 = time.time()
        resp = await engine.answer(req)
        t2 = time.time()
        print(f"Answer took {t2 - t1:.4f} seconds")
    return {
        "answer": resp.answer,
        "answer_result": json.dumps(
            resp.model_dump(),
            ensure_ascii=False,
            indent=2,
        ),
    }


if __name__ == "__main__":
    # Classic English question: What is the capital of France?
    out = anyio.run(
        main,
        "What is the capital of France?",
        False,
        None,
    )
    print(out["answer_result"])
    print(f"\nAnswer: {out['answer']}")
