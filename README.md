
# SerpSage

SerpSage is an async-only SERP utility built on top of SearxNG. It fetches results,
ranks/filters them, optionally crawls top pages to enrich the context, and can produce an AI overview with citations.

- **Config**: `AppSettings` (Pydantic), loaded via `load_settings()` from YAML/JSON + env overrides.
- **Engine**: `Engine` orchestrates the async pipeline and returns structured results.

## Configuration

Config file: `serpsage.yaml` (YAML/JSON supported).

Environment variables:
- `SERPSAGE_CONFIG_PATH`: path to JSON/YAML config (default: `serpsage.yaml`)
- `SEARXNG_BASE_URL`: overrides `provider.searxng.base_url`
- `SEARCH_API_KEY`: overrides `provider.searxng.api_key`
- `OPENAI_API_KEY`: fallback for `overview.models[*].api_key` where `backend=openai` and YAML value is empty
- `OPENAI_BASE_URL`: fallback for `overview.models[*].base_url` where `backend=openai` and YAML value is empty
- `GEMINI_API_KEY`: fallback for `overview.models[*].api_key` where `backend=gemini` and YAML value is empty
- `GEMINI_BASE_URL`: fallback for `overview.models[*].base_url` where `backend=gemini` and YAML value is empty

Note: when using the default `base_url`, `SEARCH_API_KEY` is required. This is enforced at request time.

Componentized config shape:
- Each component selects implementation via `backend` (for example: `provider.backend`, `rank.backend`, `enrich.fetch.backend`, `enrich.extractor.backend`, `cache.backend`).
- Shared HTTP transport settings live under top-level `http` and are reused by provider/fetch/overview.
- `cache` and `overview` keep an `enabled` switch.
- `overview` uses `overview.use_model` to select an entry in `overview.models[]`; each model row declares `backend` and per-model LLM options.
- Backend-specific options live under component sub-blocks (for example `enrich.fetch.playwright`, `enrich.fetch.quality_gate`, `rank.blend.providers`).
- Heuristic ranking and score normalization are configured under `rank.heuristic` (there is no separate `rank.normalization` block).
- `enrich.fetch` follows fail-fast validation (`extra=forbid`) to prevent stale/ignored keys.
- See `src/search_config_example.yaml` for a full reference.

Overview optional dependencies:
- Install `serpsage[overview]` to enable OpenAI/Gemini overview backends.

Score filtering:
- `pipeline.min_score` (default `0.5`) applies to ranked results.
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
- `low|medium|high`: crawls top-scoring pages, auto-switches between HTTP and Playwright rendering, extracts main content as clean Markdown, performs semantic chunking, and appends best chunks into each result (`result.page.chunks`)

Depth presets and fetch/chunk defaults are configurable under top-level `enrich` in settings YAML.

Enrich output now includes:
- `result.page.markdown`: cleaned main-content markdown
- `result.page.content_kind`: `html|pdf|text|binary`
- `result.page.fetch_mode`: `httpx|curl_cffi|playwright`
