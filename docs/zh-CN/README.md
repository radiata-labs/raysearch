![cover-v5-optimized](../../images/GitHub_README.png)

<div align="center">
  <a href="../../README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文文件" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../zh-CN/README.md"><img alt="简体中文文件" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語のREADME" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
</div>

# RaySearch

RaySearch 是一个异步优先的搜索编排引擎，用于在多个提供商、爬虫、提取器、排序器和 LLM 后端之上构建 AI 概览式工作流。

它提供四个高层管道：

- `search`：多提供商检索，可选抓取和重排序阶段
- `fetch`：页面爬取、提取、摘要、概览生成和相关链接
- `answer`：搜索加带引用的答案生成
- `research`：多轮研究报告生成，包含综合和结构化输出

## 为什么选择 RaySearch

- 基于组件的架构，支持可插拔的提供商、爬虫、提取器、排序器、缓存和 LLM 客户端
- 纯异步运行时，单一 `Engine` 入口点
- YAML/JSON 设置加载器，支持环境变量注入提供商和模型密钥
- 内置追踪和计量输出，便于可观测性
- 专为搜索密集型和研究密集型代理工作流设计，而非纯聊天场景

## 安装

核心安装：

```bash
uv pip install raysearch
```

常用完整安装：

```bash
uv pip install "raysearch[extract,extract_pdf,crawl,rank,cache,api,overview,tracking]"
```

使用 Playwright 爬取时，需单独安装浏览器二进制：

```bash
playwright install
```

## 公开 API

```python
from raysearch import Engine, SearchRequest, load_settings
```

主要入口点：

- `load_settings(path=None, env=None)`
- `Engine.from_settings(setting_file=None, *, settings=None, overrides=None)`
- `await engine.search(request)`
- `await engine.fetch(request)`
- `await engine.answer(request)`
- `await engine.research(request)`

## 快速开始

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

## 配置

RaySearch 按以下顺序加载设置：

1. 传递给 `load_settings(...)` 的显式 `path`
2. `RAYSEARCH_CONFIG_PATH` 环境变量
3. `raysearch.yaml` 文件
4. 代码内默认值

主要配置组：

- `components`：provider、crawl、extract、rank、llm、cache、tracking、metering、http 和速率限制
- `telemetry`：追踪和计量发射器行为
- `search`：搜索模式配置和查询扩展行为
- `fetch`：提取、摘要和概览调优
- `answer`：规划和生成模型选择
- `research`：报告生成预算和模型路由
- `runner`：并发和队列限制

组件族采用简单的默认加实例结构：

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

参考配置：

- `demo/search_config_example.yaml`

## 提供商与管道

内置提供商覆盖：

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
- `blend` 用于组合多个提供商

内置管道支持：

- 搜索结果扩展和重排序
- Markdown 优先的抓取提取
- 摘要生成和页面概览综合
- 带引用的答案生成
- 多轮研究报告生成

## 环境变量

加载器在 `AppSettings.runtime_env` 中保留完整的进程环境，组件配置模型按需从中提取值。

常用示例：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`
- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- 提供商特定覆盖，如 `GITHUB_TOKEN` 或 `SEARXNG_BASE_URL`

## 追踪与计量

追踪和计量独立于请求管道配置。

默认产物名称遵循包名：

- 追踪 JSONL：`.raysearch_tracking.jsonl`
- 计量 JSONL：`.raysearch_metering.jsonl`
- 计量 SQLite：`.raysearch_metering.sqlite3`
- 缓存 SQLite：`.raysearch_cache.sqlite3`

## 开发

仓库包含可运行的示例：

- `demo/search.py`
- `demo/fetch.py`
- `demo/answer.py`
- `demo/research.py`

示例配置：

- `demo/search_config_example.yaml`

## 注意事项

- `search.mode` 支持 `fast`、`auto` 和 `deep`
- RaySearch 仅支持异步
- 组件发现从 `raysearch.components` 加载
- JS 密集型爬取需要 Playwright 和已安装的浏览器