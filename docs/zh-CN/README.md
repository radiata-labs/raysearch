![cover-v5-optimized](../../images/GitHub_README.png)

<p align="center">
  <a href="../../README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文文件" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../zh-CN/README.md"><img alt="简体中文文件" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語的README" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
</p>

# RaySearch

RaySearch 是一个异步优先的搜索编排引擎，用于 AI 概览式工作流。可以作为：

- Python 包使用 `Engine`
- 个人 HTTP API 服务
- Docker 或 Docker Compose 部署

## 核心功能

- `search`：多提供商检索，可选抓取和重排序阶段
- `fetch`：页面爬取、提取、摘要、概览生成和相关链接
- `answer`：带引用的依据性答案生成
- `research`：多轮研究报告生成，包含综合和结构化输出

## 使用 Docker Compose 启动

```bash
git clone <repo-url>
cd google-ai-overview-api/docker
cp .env.example .env
```

编辑 `.env` 设置你的值。至少需要：

- 设置 `RAYSEARCH_API_KEY`
- 如需 `answer`、`research` 或 LLM 驱动的概览功能，设置 `OPENAI_API_KEY`

启动服务：

```bash
docker compose up -d --build
```

Compose 配置读取：

- `docker/.env` 中的环境变量
- `docker/raysearch.example.yaml` 中的服务配置

如需单独配置文件，复制 `docker/raysearch.example.yaml`，在 `.env` 中设置 `RAYSEARCH_CONFIG_FILE` 指向它。

## 使用 uv 启动

从仓库根目录：

```bash
uv run --extra service --env-file docker/.env raysearch-api --config docker/raysearch.example.yaml
```

或从 `docker` 目录内：

```bash
uv run --project .. --extra service --env-file .env raysearch-api --config raysearch.example.yaml
```

## Python 包使用

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

## 个人 API

个人 API 提供：

- `GET /healthz`
- `POST /v1/search`
- `POST /v1/fetch`
- `POST /v1/answer`
- `POST /v1/research`

使用 `api.bearer_token` 或 `RAYSEARCH_API_KEY` 中的单一共享静态 bearer token。此服务面向单用户或私有可信环境。