from __future__ import annotations

from typing import TYPE_CHECKING

import requests

from .config import DEFAULT_BASE_URL, SearchConfig

if TYPE_CHECKING:
    from collections.abc import Mapping


class SearxngClient:
    """HTTP client for SearxNG search.

    This client only fetches raw results. Post-processing lives in SearchPipeline.
    """

    config: SearchConfig
    session: requests.Session

    def __init__(
        self,
        config: SearchConfig,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()

    def search(self, query: str, *, params: Mapping[str, object] | None = None) -> dict:
        """Search via SearxNG and return raw JSON."""

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        if (
            self.config.searxng.base_url == DEFAULT_BASE_URL
            and not self.config.searxng.search_api_key
        ):
            raise ValueError(
                "SEARCH_API_KEY is required when using the default base_url."
            )

        payload: dict[str, str] = {"q": query, "format": "json"}
        if params:
            payload.update({k: str(v) for k, v in params.items()})

        response = self.session.get(
            self.config.searxng.base_url,
            params=payload,
            headers=self.config.searxng.build_headers(),
            timeout=self.config.searxng.timeout,
            allow_redirects=self.config.searxng.allow_redirects,
        )

        if response.is_redirect or response.status_code in {301, 302, 303, 307, 308}:
            raise RuntimeError(
                f"Redirected to: {response.headers.get('location')} (access not passed)"
            )

        response.raise_for_status()
        return response.json()


__all__ = ["SearxngClient"]
