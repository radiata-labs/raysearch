from __future__ import annotations

import anyio

from serpsage import FetchRequestBase, SearchRequest, load_settings
from serpsage.app.bootstrap import build_runtime
from serpsage.components import build_http_client, build_overview_client
from serpsage.models.pipeline import SearchStepContext
from serpsage.steps.search.expand import SearchExpandStep

QUERIES = [
    "what is python asyncio",
    "latest ai benchmark papers 2026",
    "kubernetes crashloopbackoff troubleshooting",
    "\u5bf9\u6bd4 FastAPI \u548c Django \u7684\u6027\u80fd",
    "\u4eca\u5929\u7f8e\u8054\u50a8\u5229\u7387\u51b3\u8bae\u662f\u4ec0\u4e48",
    "\u6771\u4eac \u89b3\u5149 \u304a\u3059\u3059\u3081 3\u65e5\u9593",
    "OpenAI API structured output json schema example",
]


async def main() -> None:
    settings = load_settings("src/search_config_example.yaml")
    settings.search.deep.expansion_model = "kimi-k2-turbo-preview"
    settings.search.deep.llm_max_queries = 2
    settings.search.deep.max_expanded_queries = 6

    rt = build_runtime(settings=settings)
    http = build_http_client(rt=rt)
    llm = build_overview_client(rt=rt, http=http)
    step = SearchExpandStep(rt=rt, llm=llm)

    async with step:
        for idx, query in enumerate(QUERIES, 1):
            req = SearchRequest(
                query=query,
                depth="deep",
                fetchs=FetchRequestBase(content=True, abstracts=False, overview=False),
            )
            ctx = SearchStepContext(
                settings=settings,
                request=req,
                request_id=f"live-expand-eval-{idx}",
            )
            out = await step.run(ctx)

            print("\n" + "=" * 90)
            print(f"CASE {idx}: {query}")
            print("- aborted:", out.deep.aborted)
            print("- jobs:", len(out.deep.query_jobs))
            if out.errors:
                for err in out.errors:
                    print("  * error:", err.code, "|", err.message)
            for job in out.deep.query_jobs:
                print(f"  - [{job.source:7}] w={job.weight:.2f}  {job.query}")


if __name__ == "__main__":
    anyio.run(main)
