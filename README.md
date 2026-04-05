![cover-v5-optimized](./images/GitHub_README.png)

<p align="center">
  <a href="./API.md">API Documentation</a> ·
  <a href="./docs/zh-TW/API.md">API 文檔</a> ·
  <a href="./docs/zh-CN/API.md">API 文档</a> ·
  <a href="./docs/ja-JP/API.md">API ドキュメント</a>
</p>

<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./docs/zh-TW/README.md"><img alt="Traditional Chinese README" src="https://img.shields.io/badge/Traditional_Chinese-d9d9d9"></a>
  <a href="./docs/zh-CN/README.md"><img alt="Simplified Chinese README" src="https://img.shields.io/badge/Simplified_Chinese-d9d9d9"></a>
  <a href="./docs/ja-JP/README.md"><img alt="Japanese README" src="https://img.shields.io/badge/Japanese-d9d9d9"></a>
</p>

# RaySearch

RaySearch is an async-first search orchestration engine for AI-overview style workflows. It can be used as:

- a Python package with `Engine`
- a personal HTTP API service
- a Docker or Docker Compose deployment

## Core Capabilities

- `search`: multi-provider retrieval with optional fetch and rerank stages
- `fetch`: page crawling, extraction, abstracts, overview generation, and related links
- `answer`: grounded answer generation with citations
- `research`: multi-round research reports with synthesis and structured output

## Start With Docker Compose

```bash
git clone https://github.com/radiata-labs/raysearch.git
cd raysearch/docker
cp .env.example .env
```

Then edit `.env` with your own values. At minimum:

- set `RAYSEARCH_API_KEY`
- set `OPENAI_API_KEY` if you want `answer`, `research`, or LLM-powered overview features

Start the service:

```bash
docker compose up -d
```

For development with hot reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up
```

The compose setup reads:

- environment from `docker/.env`
- service config from `docker/raysearch.example.yaml`

If you want a separate config file, copy `docker/raysearch.example.yaml` and point `RAYSEARCH_CONFIG_FILE` to it in `.env`.

## Start With uv

From the repository root:

```bash
uv run --extra service --env-file docker/.env raysearch-api --config docker/raysearch.example.yaml
```

Or from inside the `docker` directory:

```bash
uv run --project .. --extra service --env-file .env raysearch-api --config raysearch.example.yaml
```

## Python Package Usage

```python
from raysearch import Engine, SearchRequest

async def main() -> None:
    async with Engine.from_settings("docker/raysearch.example.yaml") as engine:
        response = await engine.search(
            SearchRequest(
                query="latest multimodal model papers",
                user_location="US",
                mode="deep",
                max_results=8,
                fetchs={"content": True},
            )
        )
        print(response.results)
```

## Personal API

The personal API exposes:

- `GET /healthz`
- `POST /v1/search`
- `POST /v1/fetch`
- `POST /v1/answer`
- `POST /v1/research`

It uses one shared static bearer token from `api.bearer_token` or `RAYSEARCH_API_KEY`. This service is intended for a single user or a private trusted environment.
