![cover-v5-optimized](../../images/GitHub_README.png)

<p align="center">
  <a href="../../README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文文件" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../zh-CN/README.md"><img alt="简体中文文件" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語のREADME" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
</p>

# RaySearch

RaySearch は AI オーバービュー形式のワークフロー向けの非同期ファースト検索オーケストレーションエンジンです。以下の形式で利用可能：

- Python パッケージ（`Engine`）
- パーソナル HTTP API サービス
- Docker または Docker Compose デプロイ

## コア機能

- `search`：複数プロバイダー検索、オプションのフェッチとリランキングステージ
- `fetch`：ページクローリング、抽出、要約、オーバービュー生成、関連リンク
- `answer`：引用付きの根拠ベース回答生成
- `research`：統合と構造化出力を含む多回合計レポート生成

## Docker Compose で始める

```bash
git clone <repo-url>
cd google-ai-overview-api/docker
cp .env.example .env
```

`.env` を編集して値を設定。最低限：

- `RAYSEARCH_API_KEY` を設定
- `answer`、`research`、LLM駆動オーバービュー機能を使用する場合は `OPENAI_API_KEY` を設定

サービスを起動：

```bash
docker compose up -d
```

開発モード（ホットリロード）：

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up
```

Compose 設定は以下を読み込み：

- `docker/.env` からの環境変数
- `docker/raysearch.example.yaml` からのサービス設定

別の設定ファイルを使用する場合、`docker/raysearch.example.yaml` をコピーし、`.env` の `RAYSEARCH_CONFIG_FILE` でそのファイルを指定。

## uv で始める

リポジトリルートから：

```bash
uv run --extra service --env-file docker/.env raysearch-api --config docker/raysearch.example.yaml
```

または `docker` ディレクトリ内から：

```bash
uv run --project .. --extra service --env-file .env raysearch-api --config raysearch.example.yaml
```

## Python パッケージの使用

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

## パーソナル API

パーソナル API は以下を公開：

- `GET /healthz`
- `POST /v1/search`
- `POST /v1/fetch`
- `POST /v1/answer`
- `POST /v1/research`

`api.bearer_token` または `RAYSEARCH_API_KEY` から単一の共有静的ベアラトークンを使用。このサービスは単一ユーザーまたはプライベート信頼環境向けです。