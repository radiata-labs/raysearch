# SPDX-License-Identifier: AGPL-3.0-or-later
"""This module implements a Wikidata provider based on the SPARQL query service.

The request shape follows the successful SearXNG approach:

- it searches entities through ``SERVICE wikibase:mwapi``
- it executes against the official SPARQL endpoint
- it resolves labels through ``SERVICE wikibase:label``
- it enriches the matched entity with official site, Wikipedia article, and
  a focused set of company and people attributes

Configuration
=============

Example configuration in this project:

.. code:: yaml

   wikidata:
     enabled: true
     base_url: https://query.wikidata.org/sparql
     user_agent: serpsage-wikidata-provider/1.0
     results_per_page: 5

Notes
=====

- The provider uses SPARQL POST requests rather than the Wikidata REST search
  endpoint because richer attribute extraction is easier there.
- It is optimized for entity-centric lookup, especially companies, people, and
  organizations.
- Result URLs prefer official sites and Wikipedia articles before falling back
  to the raw Wikidata entity URL.
"""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import httpx
from pydantic import field_validator

from serpsage.components.http.base import HttpClientBase
from serpsage.components.provider.base import (
    ProviderConfigBase,
    ProviderMeta,
    SearchProviderBase,
)
from serpsage.dependencies import Depends
from serpsage.models.components.provider import SearchProviderResult
from serpsage.utils import clean_whitespace

_DEFAULT_WIKIDATA_BASE_URL = "https://query.wikidata.org/sparql"
_DEFAULT_WIKIDATA_USER_AGENT = "serpsage-wikidata-provider/1.0"
_DEFAULT_WIKIDATA_RESULTS_PER_PAGE = 5
_MAX_WIKIDATA_RESULTS_PER_PAGE = 10
_WIKIDATA_QUERY_TEMPLATE = """
SELECT
  ?item
  ?itemLabel
  ?itemDescription
  (GROUP_CONCAT(DISTINCT STR(?officialWebsite); separator=" | ") AS ?officialWebsites)
  (GROUP_CONCAT(DISTINCT STR(?articleLocal); separator=" | ") AS ?articleLocals)
  (GROUP_CONCAT(DISTINCT STR(?articleEn); separator=" | ") AS ?articleEns)
  (GROUP_CONCAT(DISTINCT ?instanceOfLabel; separator=" | ") AS ?instanceOfLabels)
  (GROUP_CONCAT(DISTINCT ?occupationLabel; separator=" | ") AS ?occupationLabels)
  (GROUP_CONCAT(DISTINCT ?countryLabel; separator=" | ") AS ?countryLabels)
  (GROUP_CONCAT(DISTINCT ?headquartersLabel; separator=" | ") AS ?headquartersLabels)
  (GROUP_CONCAT(DISTINCT ?employerLabel; separator=" | ") AS ?employerLabels)
  (GROUP_CONCAT(DISTINCT ?ceoLabel; separator=" | ") AS ?ceoLabels)
  (GROUP_CONCAT(DISTINCT ?founderLabel; separator=" | ") AS ?founderLabels)
  (GROUP_CONCAT(DISTINCT STR(?inception); separator=" | ") AS ?inceptions)
  (GROUP_CONCAT(DISTINCT STR(?birthDate); separator=" | ") AS ?birthDates)
  (GROUP_CONCAT(DISTINCT STR(?deathDate); separator=" | ") AS ?deathDates)
  (GROUP_CONCAT(DISTINCT STR(?twitter); separator=" | ") AS ?twitters)
  (GROUP_CONCAT(DISTINCT STR(?mastodon); separator=" | ") AS ?mastodons)
WHERE {
  SERVICE wikibase:mwapi {
    bd:serviceParam wikibase:endpoint "www.wikidata.org" .
    bd:serviceParam wikibase:api "EntitySearch" .
    bd:serviceParam wikibase:limit "%LIMIT%" .
    bd:serviceParam mwapi:search "%QUERY%" .
    bd:serviceParam mwapi:language "%LANGUAGE%" .
    ?item wikibase:apiOutputItem mwapi:item .
  }
  hint:Prior hint:runFirst "true" .

  OPTIONAL { ?item wdt:P856 ?officialWebsite . }
  OPTIONAL {
    ?articleLocal schema:about ?item ;
      schema:inLanguage "%LANGUAGE%" ;
      schema:isPartOf <https://%LANGUAGE%.wikipedia.org/> .
  }
  OPTIONAL {
    ?articleEn schema:about ?item ;
      schema:inLanguage "en" ;
      schema:isPartOf <https://en.wikipedia.org/> .
  }
  OPTIONAL { ?item wdt:P31 ?instanceOf . }
  OPTIONAL { ?item wdt:P106 ?occupation . }
  OPTIONAL { ?item wdt:P17 ?country . }
  OPTIONAL { ?item wdt:P159 ?headquarters . }
  OPTIONAL { ?item wdt:P108 ?employer . }
  OPTIONAL { ?item wdt:P169 ?ceo . }
  OPTIONAL { ?item wdt:P112 ?founder . }
  OPTIONAL { ?item wdt:P571 ?inception . }
  OPTIONAL { ?item wdt:P569 ?birthDate . }
  OPTIONAL { ?item wdt:P570 ?deathDate . }
  OPTIONAL { ?item wdt:P2002 ?twitter . }
  OPTIONAL { ?item wdt:P4033 ?mastodon . }

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "%LABEL_LANGUAGES%" .
  }
}
GROUP BY ?item ?itemLabel ?itemDescription
"""


class WikidataProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "wikidata"

    base_url: str = _DEFAULT_WIKIDATA_BASE_URL
    user_agent: str = _DEFAULT_WIKIDATA_USER_AGENT
    results_per_page: int = _DEFAULT_WIKIDATA_RESULTS_PER_PAGE

    @field_validator("user_agent")
    @classmethod
    def _normalize_user_agent(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("results_per_page")
    @classmethod
    def _validate_results_per_page(cls, value: int) -> int:
        size = int(value)
        if size <= 0:
            raise ValueError("wikidata results_per_page must be > 0")
        if size > _MAX_WIKIDATA_RESULTS_PER_PAGE:
            raise ValueError(
                f"wikidata results_per_page must be <= {_MAX_WIKIDATA_RESULTS_PER_PAGE}"
            )
        return size

    @classmethod
    @override
    def inject_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> dict[str, Any]:
        payload = dict(raw)
        if env.get("WIKIDATA_BASE_URL"):
            payload["base_url"] = env["WIKIDATA_BASE_URL"]
        return payload


class WikidataProvider(
    SearchProviderBase[WikidataProviderConfig],
    meta=ProviderMeta(
        name="wikidata",
        website="https://www.wikidata.org/",
        description="Structured entity search across Wikidata for people, companies, organizations, and linked reference pages.",
        preference="Prefer people, company, organization, biography, corporate identity, and entity disambiguation queries where structured metadata helps.",
        categories=["reference", "entities"],
    ),
):
    http: HttpClientBase = Depends()

    @override
    async def _asearch(
        self,
        *,
        query: str,
        limit: int | None = None,
        language: str = "",
        location: str = "",
        moderation: bool = True,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
        **_kwargs: Any,
    ) -> list[SearchProviderResult]:
        normalized_query = clean_whitespace(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        language = self._resolve_language(language)
        page_size = self._coerce_page_size(
            limit if limit is not None else self.config.results_per_page
        )
        sparql = self._build_query(
            query=normalized_query,
            language=language,
            page_size=page_size,
        )
        resp = await self.http.client.post(
            str(self.config.base_url),
            data={"query": sparql},
            headers=self._build_headers(),
            timeout=httpx.Timeout(self.config.timeout_s),
            follow_redirects=bool(self.config.allow_redirects),
        )
        resp.raise_for_status()
        return self._parse_results(resp.json())

    def _build_headers(self) -> dict[str, str]:
        headers = dict(self.config.headers or {})
        headers["Accept"] = "application/sparql-results+json"
        headers["User-Agent"] = (
            clean_whitespace(str(self.config.user_agent or ""))
            or _DEFAULT_WIKIDATA_USER_AGENT
        )
        return headers

    def _build_query(self, *, query: str, language: str, page_size: int) -> str:
        label_languages = f"{language},en"
        return (
            _WIKIDATA_QUERY_TEMPLATE.replace(
                "%QUERY%", self._escape_sparql_string(query)
            )
            .replace("%LANGUAGE%", language)
            .replace("%LABEL_LANGUAGES%", label_languages)
            .replace("%LIMIT%", str(page_size))
        )

    def _parse_results(self, payload: Any) -> list[SearchProviderResult]:
        bindings = (
            payload.get("results", {}).get("bindings", [])
            if isinstance(payload, dict)
            else []
        )
        results: list[SearchProviderResult] = []
        seen_urls: set[str] = set()
        for binding in bindings if isinstance(bindings, list) else []:
            if not isinstance(binding, dict):
                continue
            values = {
                key: clean_whitespace(str(raw.get("value") or ""))
                for key, raw in binding.items()
                if isinstance(raw, dict)
            }
            url = self._resolve_entity_url(values)
            title = clean_whitespace(values.get("itemLabel", ""))
            if not url or not title:
                continue
            key = url.casefold()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            results.append(
                SearchProviderResult(
                    url=url,
                    title=title,
                    snippet=self._build_snippet(values),
                    engine=self.config.name,
                )
            )
        return results

    def _resolve_entity_url(self, values: dict[str, str]) -> str:
        for field_name in ("officialWebsites", "articleLocals", "articleEns", "item"):
            candidates = self._split_grouped_values(values.get(field_name, ""))
            if candidates:
                return candidates[0]
        return ""

    def _build_snippet(self, values: dict[str, str]) -> str:
        parts: list[str] = []
        description = clean_whitespace(values.get("itemDescription", ""))
        if description:
            parts.append(description)

        facts = [
            ("Type", self._split_grouped_values(values.get("instanceOfLabels", ""))),
            (
                "Occupation",
                self._split_grouped_values(values.get("occupationLabels", "")),
            ),
            ("Country", self._split_grouped_values(values.get("countryLabels", ""))),
            (
                "Headquarters",
                self._split_grouped_values(values.get("headquartersLabels", "")),
            ),
            ("Employer", self._split_grouped_values(values.get("employerLabels", ""))),
            ("CEO", self._split_grouped_values(values.get("ceoLabels", ""))),
            ("Founder", self._split_grouped_values(values.get("founderLabels", ""))),
            ("Inception", self._split_grouped_values(values.get("inceptions", ""))),
            ("Born", self._split_grouped_values(values.get("birthDates", ""))),
            ("Died", self._split_grouped_values(values.get("deathDates", ""))),
        ]
        for label, raw_values in facts:
            compact_values = [value for value in raw_values if value]
            if compact_values:
                parts.append(f"{label}: {', '.join(compact_values[:3])}")

        twitter_handles = self._split_grouped_values(values.get("twitters", ""))
        if twitter_handles:
            parts.append(f"Twitter: {', '.join(twitter_handles[:3])}")
        mastodon_handles = self._split_grouped_values(values.get("mastodons", ""))
        if mastodon_handles:
            parts.append(f"Mastodon: {', '.join(mastodon_handles[:2])}")
        return clean_whitespace(" / ".join(parts))

    def _split_grouped_values(self, value: str) -> list[str]:
        return [
            token
            for token in (
                clean_whitespace(part) for part in clean_whitespace(value).split("|")
            )
            if token
        ]

    def _resolve_language(self, language: str) -> str:
        if not language:
            return "en"
        base_language = language.split("-", 1)[0].lower()
        return base_language or "en"

    def _escape_sparql_string(self, value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )

    def _coerce_page_size(self, value: Any) -> int:
        try:
            size = int(value)
        except Exception:
            return int(self.config.results_per_page)
        return max(1, min(_MAX_WIKIDATA_RESULTS_PER_PAGE, size))


__all__ = ["WikidataProvider", "WikidataProviderConfig"]
