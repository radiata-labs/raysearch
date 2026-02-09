
# SerpSage (Async-only 3.0)

SerpSage is an async-only SERP utility built on top of SearxNG. It fetches results,
ranks/filters them, optionally crawls top pages to enrich the context, and can produce an AI overview with citations.

- **Config**: `AppSettings` (Pydantic), loaded via `load_settings()` from YAML/JSON + env overrides.
- **Engine**: `Engine` orchestrates the async pipeline and returns structured results.

## Configuration

Config file: `serpsage.yaml` (YAML/JSON supported).

Environment variables:
- `SERPSAGE_CONFIG_PATH`: path to JSON/YAML config (default: `serpsage.yaml`)
- `SEARXNG_BASE_URL`: overrides `searxng.base_url`
- `SEARCH_API_KEY`: overrides `searxng.search_api_key`

Note: when using the default `base_url`, `SEARCH_API_KEY` is required. This is enforced at request time.

Score filtering:
- `score_filter.min_score` (default `0.5`) applies to both ranked results and web chunks.
- Items with `score == 0.0` are always dropped.

## Python usage (async-only)

```python
from serpsage import Engine, SearchRequest, load_settings

settings = load_settings()

req = SearchRequest(query="example query", depth="simple", max_results=16)

async with Engine.from_settings(settings) as engine:
    resp = await engine.run(req)
```

## Search Depth (Web Crawl Enrichment)

`depth` is a runtime parameter:
- `simple`: only uses SearxNG snippets (default)
- `low|medium|high`: crawls the top-scoring pages, extracts full page text, chunks it with overlap, scores chunks, then appends best chunks into each result (`result.page.chunks`)

Depth presets and fetch/chunk defaults are configurable under top-level `enrich` in settings YAML.
