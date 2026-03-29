![cover-v5-optimized](./images/GitHub_README.png)

<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./docs/zh-TW/README.md"><img alt="繁體中文文件" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="./docs/zh-CN/README.md"><img alt="简体中文文件" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="./docs/ja-JP/README.md"><img alt="日本語のREADME" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
</p>

#

RaySearch is an async-first search orchestration engine for building AI-overview style workflows on top of multiple providers, crawlers, extractors, rankers, and LLM backends.

It exposes four high-level pipelines:

- `search`: multi-provider retrieval with optional fetch and rerank stages
- `fetch`: page crawling, extraction, abstracting, overview generation, and related links
- `answer`: search plus grounded answer generation with citations
- `research`: multi-round research reports with synthesis and structured output

## Why RaySearch

- Component-based architecture with pluggable providers, crawlers, extractors, rankers, caches, and LLM clients
- Async-only runtime with a single `Engine` entry point
- YAML/JSON settings loader plus environment injection for provider and model secrets
- Built-in tracking and metering sinks for observability
- Designed for search-heavy and research-heavy agent workflows rather than chat-only use cases

## Installation

Core install:

```bash
uv pip install raysearch
```

Common full install:

```bash
uv pip install "raysearch[extract,extract_pdf,crawl,rank,cache,api,overview,tracking]"
```

When using Playwright-based crawling, install browser binaries separately:

```bash
playwright install
```

## Public API

```python
from raysearch import Engine, SearchRequest, load_settings
```

Primary entry points:

- `load_settings(path=None, env=None)`
- `Engine.from_settings(setting_file=None, *, settings=None, overrides=None)`
- `await engine.search(request)`
- `await engine.fetch(request)`
- `await engine.answer(request)`
- `await engine.research(request)`

## Quick Start

```python
from raysearch import Engine, SearchRequest

async def main() -> None:
    async with Engine.from_settings("demo/search_config_example.yaml") as engine:
        response = await engine.search(
            SearchRequest(
                query="latest multimodal model papers",
                mode="deep",
                max_results=8,
            )
        )
        for item in response.results:
            print(item.title, item.url)
```

## Configuration

RaySearch loads settings in this order:

1. Explicit `path` passed to `load_settings(...)`
2. `RAYSEARCH_CONFIG_PATH`
3. `raysearch.yaml`
4. In-code defaults

The main configuration groups are:

- `components`: provider, crawl, extract, rank, llm, cache, tracking, metering, http, and rate limiting
- `telemetry`: tracking and metering emitter behavior
- `search`: search-mode profiles and query-expansion behavior
- `fetch`: extraction, abstract, and overview tuning
- `answer`: planning and generation model selection
- `research`: report-generation budgets and model routing
- `runner`: concurrency and queue limits

Component families use a simple default-plus-instance shape:

```yaml
components:
  provider:
    default: google
    google:
      enabled: true
      cookies:
        CONSENT: "YES+"
    duckduckgo:
      enabled: true
      base_url: https://html.duckduckgo.com/html
      allow_redirects: false
```

Reference configuration:

- `demo/search_config_example.yaml`

## Providers And Pipelines

Built-in provider coverage includes:

- `google`
- `google_news`
- `duckduckgo`
- `searxng`
- `github`
- `reddit`
- `reuters`
- `openalex`
- `semantic_scholar`
- `wikidata`
- `wikipedia`
- `arxiv`
- `marginalia`
- `blend` for combining providers

Built-in pipeline support includes:

- Search result expansion and reranking
- Markdown-first fetch extraction
- Abstract generation and page overview synthesis
- Citation-grounded answer generation
- Multi-round research report generation

## Environment Variables

The loader preserves the full process environment in `AppSettings.runtime_env`, and component config models pull values from there as needed.

Common examples:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`
- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- Provider-specific overrides such as `GITHUB_TOKEN` or `SEARXNG_BASE_URL`

## Tracking And Metering

Tracking and metering are configured independently from the request pipelines.

Default artifact names now follow the package name:

- tracking JSONL: `.raysearch_tracking.jsonl`
- metering JSONL: `.raysearch_metering.jsonl`
- metering SQLite: `.raysearch_metering.sqlite3`
- cache SQLite: `.raysearch_cache.sqlite3`

## Development

The repo includes runnable demos:

- `demo/search.py`
- `demo/fetch.py`
- `demo/answer.py`
- `demo/research.py`

Example settings:

- `demo/search_config_example.yaml`

## Notes

- `search.mode` supports `fast`, `auto`, and `deep`
- RaySearch is async-only
- Component discovery loads from `raysearch.components`
- JS-heavy crawling requires Playwright plus installed browsers
