![cover-v5-optimized](../../images/GitHub_README.png)

<div align="center">
  <a href="../../API.md">API Documentation</a> ·
  <a href="../zh-TW/API.md">API 文檔</a> ·
  <a href="../zh-CH/API.md">API 文档</a> ·
  <a href="../ja-JP/API.md">API ドキュメント</a> ·
</div>

<p align="center">
  <a href="../../README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文文件" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../zh-CN/README.md"><img alt="简体中文文件" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語的README" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
</p>

# RaySearch

RaySearch 是一個異步優先的搜尋編排引擎，用於 AI 概覽式工作流。可以作為：

- Python 套件使用 `Engine`
- 個人 HTTP API 服務
- Docker 或 Docker Compose 部署

## 核心功能

- `search`：多提供商檢索，可選抓取和重排序階段
- `fetch`：頁面爬取、提取、摘要、概覽生成和相關連結
- `answer`：帶引用的依據性答案生成
- `research`：多輪研究報告生成，包含綜合和結構化輸出

## 使用 Docker Compose 啟動

```bash
git clone <repo-url>
cd google-ai-overview-api/docker
cp .env.example .env
```

編輯 `.env` 設定你的值。至少需要：

-設定 `RAYSEARCH_API_KEY`
- 如需 `answer`、`research` 或 LLM 驅動的概覽功能，設定 `OPENAI_API_KEY`

啟動服務：

```bash
docker compose up -d
```

開發模式（熱重載）：

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up
```

Compose 配置讀取：

- `docker/.env` 中的環境變數
- `docker/raysearch.example.yaml` 中的服務配置

如需單獨配置檔案，複製 `docker/raysearch.example.yaml`，在 `.env` 中設定 `RAYSEARCH_CONFIG_FILE` 指向它。

## 使用 uv 啟動

從倉庫根目錄：

```bash
uv run --extra service --env-file docker/.env raysearch-api --config docker/raysearch.example.yaml
```

或從 `docker` 目錄內：

```bash
uv run --project .. --extra service --env-file .env raysearch-api --config raysearch.example.yaml
```

## Python 套件使用

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

## 個人 API

個人 API 提供：

- `GET /healthz`
- `POST /v1/search`
- `POST /v1/fetch`
- `POST /v1/answer`
- `POST /v1/research`

使用 `api.bearer_token` 或 `RAYSEARCH_API_KEY` 中的單一共享靜態 bearer token。此服務面向單用戶或私有可信環境。