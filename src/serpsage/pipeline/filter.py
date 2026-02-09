from __future__ import annotations

from urllib.parse import urlparse

from serpsage.app.container import Container
from serpsage.pipeline.steps import StepContext
from serpsage.text.normalize import normalize_text
from serpsage.text.tokenize import tokenize


class FilterStep:
    def __init__(self, container: Container) -> None:
        self._c = container

    async def run(self, ctx: StepContext) -> StepContext:
        span = self._c.telemetry.start_span("step.filter")
        try:
            req = ctx.request
            profile_name, profile = ctx.settings.select_profile(
                query=req.query,
                explicit=req.profile,
            )
            ctx.profile_name = profile_name
            ctx.profile = profile

            noise_exts = {e.lower().lstrip(".") for e in (profile.noise_extensions or [])}

            query_tokens = [t for t in tokenize(req.query) if len(t) >= 2]
            ctx.scratch["query_tokens"] = query_tokens

            kept = []
            for r in ctx.results:
                if not _is_not_noise(r.url, r.title, r.snippet, r.domain, profile.noise_words, noise_exts):
                    continue
                if not _is_relevant(r.title, r.snippet, query_tokens):
                    continue
                kept.append(r)

            ctx.results = kept
            span.set_attr("n", len(kept))
            return ctx
        finally:
            span.end()


def _is_not_noise(
    url: str,
    title: str,
    snippet: str,
    domain: str,
    noise_words: list[str],
    noise_exts: set[str],
) -> bool:
    title = (title or "").strip()
    snippet = (snippet or "").strip()
    url = (url or "").strip()
    domain = (domain or "").strip()
    blob = f"{title} {snippet} {url} {domain}".lower()

    if not title and not snippet:
        return False

    if url:
        path = urlparse(url).path.lower()
        for ext in noise_exts:
            if path.endswith(f".{ext}"):
                return False

    lowered = normalize_text(blob)
    for w in noise_words or []:
        wl = normalize_text(w)
        if wl and wl in lowered:
            return False

    return not (len(title) < 2 and len(snippet) < 40)


def _is_relevant(title: str, snippet: str, query_tokens: list[str]) -> bool:
    t = (title or "").lower()
    s = (snippet or "").lower()
    return any(tok in t or tok in s for tok in query_tokens)


__all__ = ["FilterStep"]

