# SerpSage

SerpSage is an async-only SERP + page intelligence engine with two first-class paths:

- `Engine.search(req)`: query search pipeline (`search -> normalize -> filter -> dedupe -> rank -> fetch-enhance -> rerank -> optional overview`)
- `Engine.fetch(req)`: multi-URL pipeline (`prepare -> load(cache/crawl) -> extract -> optional chunk rank -> optional overview -> finalize`)

The `fetch` path is Markdown-first: output is centered on clean main-content markdown (`response.results[].content`).

## Public API

- `Engine`
- `SearchRequest` / `SearchResponse`
- `FetchRequest` / `FetchResponse`
- `load_settings`

## Configuration

Top-level settings:

- `provider`
- `http`
- `search`
- `fetch`
- `rank`
- `llm`
- `cache`
- `telemetry`

Reference file: `src/search_config_example.yaml`.

Env overrides:

- `SERPSAGE_CONFIG_PATH`
- `SEARXNG_BASE_URL`
- `SEARCH_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`

Runtime prerequisites:

- JS rendering uses Playwright. Install browsers with `playwright install`.
- Overview uses optional dependencies (`serpsage[overview]`) plus API keys.

## Usage

```python
from serpsage import (
    Engine,
    FetchChunksRequest,
    FetchContentRequest,
    FetchOverviewRequest,
    FetchRequest,
    SearchRequest,
    load_settings,
)

settings = load_settings()

async with Engine.from_settings(settings) as engine:
    search_resp = await engine.search(
        SearchRequest(query="latest ai papers", depth="medium", max_results=8)
    )
    fetch_resp = await engine.fetch(
        FetchRequest(
            urls=["https://example.com/article"],
            crawl_mode="fallback",
            crawl_timeout=2.5,
            content=FetchContentRequest(depth="medium"),
            chunks=FetchChunksRequest(query="benchmark results", top_k_chunks=3),
            overview=FetchOverviewRequest(query="benchmark results"),
        )
    )
```

## Behavior notes

- `search.depth`: `simple|low|medium|high` (kept semantics)
- `fetch`: no depth tiers, single strategy profile
- `FetchRequest` V3 fields:
  - `urls`: list of URLs, processed concurrently with stable output order
  - `crawl_mode`: `never|fallback|preferred|always`
  - `crawl_timeout`: per-URL crawler timeout in seconds
  - `content`: `bool | FetchContentRequest` (`false` hides output markdown only; internal extraction still runs)
  - `chunks`: `FetchChunksRequest | None` (controls chunking query / limits)
  - `overview`: `FetchOverviewRequest | None` (controls overview query / limits)
  - `runtime.max_links`: optional link collection cap; omitted means no links output
  - old fetch fields (`url/params/query/include_chunks/top_k_chunks/include_secondary_content`) are removed
- both `search` and `fetch` support optional overview
- fetch/extract pipeline supports JS-rendered pages, PDF text extraction, and noisy layouts with boilerplate filtering
