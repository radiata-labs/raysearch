from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from serpsage.core.workunit import WorkUnit
from serpsage.models.outcomes import FilterOutcome
from serpsage.text.normalize import normalize_text
from serpsage.text.tokenize import tokenize

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.settings.models import ProfileSettings


class Filterer(WorkUnit):
    def filter(
        self,
        *,
        query: str,
        explicit_profile: str | None,
        results: list[ResultItem],
    ) -> FilterOutcome:
        profile_name, profile = self.settings.select_profile(
            query=query, explicit=explicit_profile
        )

        noise_exts = {e.lower().lstrip(".") for e in (profile.noise_extensions or [])}
        query_tokens = [t for t in tokenize(query) if len(t) >= 2]

        kept: list[ResultItem] = []
        for r in results:
            if not self._is_not_noise(r, profile, noise_exts):
                continue
            if not self._is_relevant(r, query_tokens):
                continue
            kept.append(r)

        return FilterOutcome(
            profile_name=profile_name,
            profile=profile,
            query_tokens=query_tokens,
            results=kept,
        )

    def _is_not_noise(
        self,
        r: ResultItem,
        profile: ProfileSettings,
        noise_exts: set[str],
    ) -> bool:
        title = (r.title or "").strip()
        snippet = (r.snippet or "").strip()
        url = (r.url or "").strip()
        domain = (r.domain or "").strip()
        blob = f"{title} {snippet} {url} {domain}".lower()

        if not title and not snippet:
            return False

        if url:
            path = urlparse(url).path.lower()
            for ext in noise_exts:
                if path.endswith(f".{ext}"):
                    return False

        lowered = normalize_text(blob)
        for w in profile.noise_words or []:
            wl = normalize_text(w)
            if wl and wl in lowered:
                return False

        return not (len(title) < 2 and len(snippet) < 40)

    def _is_relevant(self, r: ResultItem, query_tokens: list[str]) -> bool:
        t = (r.title or "").lower()
        s = (r.snippet or "").lower()
        return any(tok in t or tok in s for tok in query_tokens)


__all__ = ["FilterOutcome", "Filterer"]
