# SerpSage

SerpSage is an async-only SERP + page intelligence engine with two first-class paths:

- `Engine.search(req)`: query search pipeline (`prepare -> search(prefetch) -> fetch -> finalize`)
- `Engine.fetch(req)`: multi-URL pipeline (`prepare -> load(cache/crawl) -> extract -> optional abstract rank -> optional overview -> optional subpages -> finalize`)

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
- HTML->Markdown conversion uses an internal renderer (compat backend name remains `markdownify`).

## Usage

```python
from serpsage import (
    Engine,
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOthersRequest,
    FetchOverviewRequest,
    FetchRequest,
    FetchSubpagesRequest,
    SearchRequest,
    load_settings,
)

settings = load_settings()

async with Engine.from_settings(settings) as engine:
    search_resp = await engine.search(
        SearchRequest(
            query="latest ai papers",
            depth="deep",
            max_results=8,
            fetchs={"content": True},
        )
    )
    fetch_resp = await engine.fetch(
        FetchRequest(
            urls=["https://example.com/article"],
            crawl_mode="fallback",
            crawl_timeout=2.5,
            content=FetchContentRequest(detail="standard"),
            abstracts=FetchAbstractsRequest(query="benchmark results", max_chars=2400),
            subpages=FetchSubpagesRequest(
                max_subpages=2,
                subpage_keywords="benchmark, evaluation",
            ),
            overview=FetchOverviewRequest(query="benchmark results"),
            others=FetchOthersRequest(max_links=20, max_image_links=10),
        )
    )
```

## Behavior notes

- `search.depth`: `auto|deep`
- `fetch`: no depth tiers, single strategy
- `fetch.backend`: `auto|curl_cffi|playwright` (`httpx` backend removed)
- migration: old search profile/overview settings are removed
- `FetchRequest` V4 fields:
  - `urls`: list of URLs, processed concurrently with stable output order
  - `crawl_mode`: `never|fallback|preferred|always`
  - `crawl_timeout`: per-URL crawler timeout in seconds
  - `content`: `bool | FetchContentRequest`, default `false` (`false` hides output markdown only; internal extraction still runs)
  - `FetchContentRequest.detail`: `concise|standard|full` (mapped internally to original extraction depth behavior)
  - `abstracts`: `bool | FetchAbstractsRequest`, default `false` (`true` means enabled with default config)
  - `subpages`: `FetchSubpagesRequest | None`
  - `subpages.max_subpages`: required to enable subpages; `None` means disabled
  - `subpages.subpage_keywords`: one string or comma-separated keywords for ranking links
  - `overview`: `bool | FetchOverviewRequest`, default `false` (`true` means enabled with default config)
  - `FetchAbstractsRequest.query` / `FetchOverviewRequest.query` may be `None`; ranking query falls back to `title`, then `url`
  - when `overview.query` is `None`, the LLM query is fixed as `总结`
  - `others.max_links` / `others.max_image_links`: optional link collection caps; omitted means no links output
  - if `subpages` is enabled but `others.max_links` is omitted, fetch internally collects links for subpage ranking and may hide those links in final `others.links`
  - old fetch fields (`url/params/query/include_chunks/top_k_chunks/include_secondary_content/runtime`) are removed
- `search` does not include overview generation; overview remains fetch-only
- `search` response shape: `search_depth/results/errors/telemetry`
- fetch/extract pipeline supports JS-rendered pages, PDF text extraction, and noisy layouts with boilerplate filtering
- `fetch.extract` uses an internal markdown renderer pipeline; no renderer backend toggle is exposed.
