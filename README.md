# Google AI Overview API

This repo provides a small search utility built on top of SearxNG:

- **Config**: a single `SearchConfig` (Pydantic) loaded from `search_config.yaml` (JSON/YAML).
- **Client**: `SearxngClient` only fetches raw JSON results.
- **Pipeline**: `SearchPipeline` selects a profile (auto-match) and processes/ranks results, then renders markdown.

## Configuration

Config file: `search_config.yaml` (JSON/YAML supported).

Environment variables:
- `SEARCH_CONFIG_PATH`: path to JSON/YAML config (default: `search_config.yaml`)
- `SEARCH_PROFILE`: explicit profile name (must exist; overrides auto selection)
- `SEARXNG_BASE_URL`: overrides `searxng.base_url`
- `SEARCH_API_KEY`: overrides `searxng.search_api_key`

Note: when using the default `base_url`, `SEARCH_API_KEY` is required. This is enforced at request time.

## Python usage

```python
from search_core import SearchConfig, SearchPipeline

cfg = SearchConfig.load()
engine = SearchPipeline(cfg)

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

## Search Depth (Web Crawl Enrichment)

`depth` is a runtime parameter (not stored in config):
- `simple`: only uses SearxNG snippets (default)
- `low|medium|high`: crawls the top-scoring pages, extracts full page text, chunks it with overlap, scores chunks, then appends best chunks into each result (`page_chunks`)

Depth presets and crawler/chunk/scoring defaults are configurable under top-level `web_enrichment` in `search_config.yaml`.

Chunk parameters (runtime, overrides config when provided):
- `chunk_target_chars`: approximate chunk size in characters (sentence boundaries preserved)
- `chunk_overlap_sentences`: overlap in number of sentences
- `min_chunk_chars`: minimum chunk size to keep
- `max_chunk_chars`: how many characters to show per chunk in markdown
