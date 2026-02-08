
# SerpSage

SerpSage is a small SERP utility built on top of SearxNG. It fetches results,
ranks/filters them, and optionally crawls top pages to enrich the context (AI-overview style).

- **Config**: `SearchConfig` (Pydantic), loaded from `search_config.yaml` (JSON/YAML).
- **Client**: `SearxngClient` (sync) / `AsyncSearxngClient` (async) fetch raw JSON results.
- **Core**: `Searcher` / `AsyncSearcher` process, rank, and render output.

## Configuration

Config file: `search_config.yaml` (JSON/YAML supported).

Environment variables:
- `SEARCH_CONFIG_PATH`: path to JSON/YAML config (default: `search_config.yaml`)
- `SEARCH_PROFILE`: explicit profile name (must exist; overrides auto selection)
- `SEARXNG_BASE_URL`: overrides `searxng.base_url`
- `SEARCH_API_KEY`: overrides `searxng.search_api_key`

Note: when using the default `base_url`, `SEARCH_API_KEY` is required. This is enforced at request time.

Score filtering:
- `score_filter.min_score` (default `0.5`) applies to both ranked results and web chunks.
- Items with `score == 0.0` are always dropped.

## Python usage

```python
from search_core import SearchConfig, Searcher

cfg = SearchConfig.load()
engine = Searcher(cfg)

markdown = engine.search_markdown(
    "example query",
    "simple",  # depth: simple|low|medium|high
    profile=None,  # if None: uses SEARCH_PROFILE env var or auto-match rules
    max_results=16,
    max_snippet_chars=1000,
    show_source_domain=True,
    show_source_url=False,
    show_source_engine=False,
)
```

## Async Python usage

```python
import anyio

from search_core import AsyncSearcher, SearchConfig


async def run() -> None:
    cfg = SearchConfig.load()
    async with AsyncSearcher(cfg) as engine:
        md = await engine.asearch_markdown(
            "example query",
            "high",  # depth: simple|low|medium|high
            max_results=8,
        )
        print(md)


anyio.run(run)
```

## Search Depth (Web Crawl Enrichment)

`depth` is a runtime parameter (not stored in config):
- `simple`: only uses SearxNG snippets (default)
- `low|medium|high`: crawls the top-scoring pages, extracts full page text, chunks it with overlap, scores chunks, then appends best chunks into each result (`result.page.chunks`)

Depth presets and crawler/chunk/scoring defaults are configurable under top-level `web_enrichment` in `search_config.yaml`.

Scoring strategy/weights are configured globally under `scoring.providers` in `search_config.yaml`.
`web_enrichment.select` only contains chunk-specific thresholds/penalties.

Web fetching/decoding is handled by `WebCrawler` (charset-aware best-effort decoding).

Chunk parameters (runtime, overrides config when provided):
- `chunk_target_chars`: approximate chunk size in characters (sentence boundaries preserved)
- `chunk_overlap_sentences`: overlap in number of sentences
- `min_chunk_chars`: minimum chunk size to keep
- `max_chunk_chars`: how many characters to show per chunk in markdown
