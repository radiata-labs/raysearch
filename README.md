# Google AI Overview API

This repo provides a small search utility built on top of SearxNG:

- **Config**: a single `SearchConfigFile` (Pydantic) loaded from `search_config.yaml` (JSON/YAML).
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
from search_core import SearchConfigFile, SearchPipeline

cfg = SearchConfigFile.load()
engine = SearchPipeline(cfg)

markdown = engine.search_markdown(
    "example query",
    profile=None,  # if None: uses SEARCH_PROFILE env var or auto-match rules
    max_results=16,
    max_snippet_chars=1000,
    show_source_domain=True,
    show_source_url=False,
    show_source_engine=False,
)
```

