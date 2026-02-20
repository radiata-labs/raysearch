from __future__ import annotations

import json
from typing import Any

import anyio
from dotenv import load_dotenv

from serpsage import AnswerRequest, Engine, load_settings

load_dotenv()


async def main(
    query: str,
    content: bool = True,
    json_schema: object | None = None,
) -> dict[str, Any]:
    settings = load_settings("src/search_config_example.yaml")
    req = AnswerRequest(
        query=query,
        content=content,
        json_schema=json_schema,
    )
    async with Engine.from_settings(settings) as engine:
        resp = await engine.answer(req)
    resp_dict = resp.model_dump()
    resp_dict["telemetry"]["spans"] = [
        item for item in resp_dict["telemetry"]["spans"] if "fetch" not in item["name"]
    ]
    return {
        "answer": resp.answer,
        "answer_result": json.dumps(
            resp_dict,
            ensure_ascii=False,
            indent=2,
        ),
    }


if __name__ == "__main__":
    import time

    t1 = time.time()
    out = anyio.run(main, "What is curl_cffi?", False, None)
    t2 = time.time()

    print(out["answer_result"])
    print(f"Answer: {out['answer']}")
    print(f"Answer took {t2 - t1:.2f} seconds")
