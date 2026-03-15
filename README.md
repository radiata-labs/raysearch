# SerpSage

SerpSage is an async-only SERP, fetch, answer, and research engine. Public entry points remain:

- `load_settings()`
- `Engine.from_settings(settings)`
- `Engine.search(req)`
- `Engine.fetch(req)`
- `Engine.answer(req)`
- `Engine.research(req)`

## Component System

The component layer now loads through `serpsage.components.loads` and is isolated per
`Engine.from_settings(...)` call.

- Each component owns its own `pydantic` config model.
- `src/serpsage/settings/models.py` only defines generic component-family containers plus non-component business settings.
- Builtin components self-register through metadata attached at import time.
- Each `from_settings(...)` call builds a fresh registry and component catalog.
- Raw user-declared instances are tracked separately from merged defaults.
- WorkUnit bootstrap uses direct dependency injection for settings, clock, tracking, metering, and component registry.
- `backend: Literal[...]` is removed from settings. Families now declare `default` and `instances`.
- Components with `config_optional=True` may load from merged defaults even when
  the instance was not explicitly written in the config file.

Built-in component families:

- `http`
- `provider`
- `fetch`
- `extract`
- `rank`
- `llm`
- `tracking`
- `metering`
- `cache`
- `rate_limit`

## Configuration Shape

Every component family uses the same structure:

```yaml
provider:
  default: google_main
  instances:
    google_main:
      component: google
      enabled: true
      config:
        country: US
```

Example:

```yaml
llm:
  default: router_main
  instances:
    router_main:
      component: router
      enabled: true
      config: {}
    gpt41_mini:
      component: openai
      enabled: true
      config:
        name: gpt-4.1-mini
        model: gpt-4.1-mini
        api_key: null
```

Reference config: `demo/search_config_example.yaml`.

## Environment Injection

`load_settings()` only loads the top-level file and preserves environment values in `AppSettings.runtime_env`. Component-specific env overrides are implemented by each component config model.

Examples:

- `openai` route instances can read `OPENAI_API_KEY` and `OPENAI_BASE_URL`
- `gemini` route instances can read `GEMINI_API_KEY` and `GEMINI_BASE_URL`
- `google` and `searxng` providers can read provider/search env overrides

Global loader behavior:

- explicit `path`
- `SERPSAGE_CONFIG_PATH`
- `serpsage.yaml`
- defaults

## Tracking And Metering

Tracking and metering are separate component families.

Built-in tracking components:

- emitters: `null_emitter`, `async_emitter`
- sinks: `null_sink`, `jsonl_sink`

Built-in metering components:

- emitters: `null_emitter`, `async_emitter`
- sinks: `null_sink`, `jsonl_sink`, `sqlite_sink`

Example:

```yaml
tracking:
  default: async_emitter
  async_emitter:
    enabled: true
    queue_size: 2048
    minimum_level: INFO
  jsonl_sink:
    enabled: true
    jsonl_path: .serpsage_tracking.jsonl

metering:
  default: async_emitter
  async_emitter:
    enabled: true
    queue_size: 2048
  sqlite_sink:
    enabled: true
    sqlite_db_path: .serpsage_metering.sqlite3
```

## Usage

```python
from serpsage import Engine, SearchRequest, load_settings

settings = load_settings("demo/search_config_example.yaml")

async with Engine.from_settings(settings) as engine:
    response = await engine.search(
        SearchRequest(
            query="latest ai papers",
            mode="deep",
            max_results=8,
        )
    )
```

## Notes

- `search.mode`: `fast | auto | deep`
- Fetch remains markdown-first.
- Component loading lives under `serpsage.components.loads`.
- JS rendering still requires Playwright browsers to be installed.
