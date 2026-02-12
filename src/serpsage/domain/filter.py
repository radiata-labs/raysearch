from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from serpsage.core.workunit import WorkUnit
from serpsage.text.normalize import normalize_text

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.settings.models import ProfileSettings


class Filterer(WorkUnit):
    def filter(
        self,
        results: list[ResultItem],
        *,
        query_tokens: list[str],
        profile: ProfileSettings,
    ) -> list[ResultItem]:

        noise_exts = {e.lower().lstrip(".") for e in (profile.noise_extensions or [])}

        kept: list[ResultItem] = [
            r
            for r in results
            if self._is_not_noise(r, profile, noise_exts)
            and self._is_relevant(r, query_tokens)
        ]

        return kept

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


__all__ = ["Filterer"]
