from __future__ import annotations

# mypy: disable-error-code="untyped-decorator,no-any-return"
# pyright: reportUnusedFunction=false
import argparse
import os
import secrets
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from raysearch import (
    AnswerRequest,
    AnswerResponse,
    Engine,
    FetchRequest,
    FetchResponse,
    ResearchRequest,
    ResearchResponse,
    SearchRequest,
    SearchResponse,
    load_settings,
)
from raysearch.settings.models import AppSettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from fastapi import FastAPI


def _require_fastapi() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request, status
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "FastAPI support requires the `raysearch[service]` or `raysearch[api]` extra."
        ) from exc
    return (
        Depends,
        FastAPI,
        HTTPException,
        HTTPAuthorizationCredentials,
        HTTPBearer,
        Request,
        status,
    )


def _require_uvicorn() -> Any:
    try:
        import uvicorn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The API server requires the `raysearch[service]` or `raysearch[api]` extra."
        ) from exc
    return uvicorn


def _clone_settings(
    settings: AppSettings | dict[str, Any],
) -> AppSettings | dict[str, Any]:
    if isinstance(settings, AppSettings):
        return settings.model_copy(deep=True)
    payload = dict(settings)
    if "runtime_env" not in payload:
        payload["runtime_env"] = dict(os.environ)
    return payload


def _resolve_app_settings(
    setting_file: str | None = None,
    *,
    settings: AppSettings | dict[str, Any] | None = None,
) -> AppSettings:
    if settings is None:
        raw_settings = load_settings(path=setting_file)
        return AppSettings.model_validate(raw_settings)
    normalized = _clone_settings(settings)
    if isinstance(normalized, AppSettings):
        return normalized
    return AppSettings.model_validate(normalized)


def _raise_http_401(detail: str) -> None:
    fastapi_parts = _require_fastapi()
    http_exception_cls = fastapi_parts[2]
    status = fastapi_parts[6]
    raise http_exception_cls(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _require_bearer_token(settings: AppSettings) -> str:
    token = settings.api.bearer_token.strip()
    if not token:
        raise RuntimeError(
            "API bearer token is required. Set `api.bearer_token` or `RAYSEARCH_API_KEY`."
        )
    return token


def create_api_app(
    setting_file: str | None = None,
    *,
    settings: AppSettings | dict[str, Any] | None = None,
) -> FastAPI:
    (
        depends,
        fastapi_cls,
        http_exception_cls,
        _http_authorization_credentials,
        http_bearer_cls,
        _request_cls,
        status,
    ) = _require_fastapi()
    app_settings = _resolve_app_settings(setting_file, settings=settings)
    bearer_token = _require_bearer_token(app_settings)

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        engine = Engine.from_settings(settings=app_settings)
        await engine.ainit()
        app.state.engine = engine
        app.state.app_settings = app_settings
        try:
            yield
        finally:
            await engine.aclose()

    docs_url = "/docs" if app_settings.api.enable_docs else None
    openapi_url = "/openapi.json" if app_settings.api.enable_docs else None
    app = fastapi_cls(
        title="RaySearch Personal API",
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    auth_scheme = http_bearer_cls(auto_error=False)

    async def require_auth(
        credentials: Any = depends(auth_scheme),
    ) -> None:
        if credentials is None:
            _raise_http_401("missing bearer token")
        scheme = str(credentials.scheme or "")
        token = str(credentials.credentials or "")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token,
            bearer_token,
        ):
            _raise_http_401("invalid bearer token")

    def get_engine(request: Any) -> Engine:
        engine = getattr(request.app.state, "engine", None)
        if not isinstance(engine, Engine):
            raise http_exception_cls(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="engine is not ready",
            )
        return engine

    async def _run_operation(
        operation: Callable[[], Awaitable[Any]],
        *,
        error_detail: str,
    ) -> Any:
        try:
            return await operation()
        except http_exception_cls:
            raise
        except Exception as exc:  # noqa: BLE001
            raise http_exception_cls(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_detail,
            ) from exc

    @app.get("/healthz")
    async def healthz(request: Any) -> dict[str, object]:
        return {
            "status": "ok",
            "engine_ready": isinstance(
                getattr(request.app.state, "engine", None),
                Engine,
            ),
        }

    @app.post(
        "/v1/search",
        response_model=SearchResponse,
        dependencies=[depends(require_auth)],
    )
    async def search_endpoint(
        payload: SearchRequest,
        engine: Engine = depends(get_engine),
    ) -> SearchResponse:
        return await _run_operation(
            lambda: engine.search(payload),
            error_detail="search request failed",
        )

    @app.post(
        "/v1/fetch",
        response_model=FetchResponse,
        dependencies=[depends(require_auth)],
    )
    async def fetch_endpoint(
        payload: FetchRequest,
        engine: Engine = depends(get_engine),
    ) -> FetchResponse:
        return await _run_operation(
            lambda: engine.fetch(payload),
            error_detail="fetch request failed",
        )

    @app.post(
        "/v1/answer",
        response_model=AnswerResponse,
        dependencies=[depends(require_auth)],
    )
    async def answer_endpoint(
        payload: AnswerRequest,
        engine: Engine = depends(get_engine),
    ) -> AnswerResponse:
        return await _run_operation(
            lambda: engine.answer(payload),
            error_detail="answer request failed",
        )

    @app.post(
        "/v1/research",
        response_model=ResearchResponse,
        dependencies=[depends(require_auth)],
    )
    async def research_endpoint(
        payload: ResearchRequest,
        engine: Engine = depends(get_engine),
    ) -> ResearchResponse:
        return await _run_operation(
            lambda: engine.research(payload),
            error_detail="research request failed",
        )

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the RaySearch personal API server."
    )
    parser.add_argument(
        "--config",
        dest="setting_file",
        default=None,
        help="Optional path to a YAML or JSON RaySearch config file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    app_settings = _resolve_app_settings(args.setting_file)
    _require_bearer_token(app_settings)
    uvicorn = _require_uvicorn()
    uvicorn.run(
        create_api_app(args.setting_file, settings=app_settings),
        host=app_settings.api.host,
        port=app_settings.api.port,
    )
    return 0


__all__ = ["create_api_app", "main"]
