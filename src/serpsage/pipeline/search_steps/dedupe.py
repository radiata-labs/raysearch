from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import SearchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.text.normalize import clean_whitespace
from serpsage.text.similarity import hybrid_similarity, simhash64

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import ProfileSettings

_WS_RE = re.compile(r"\s+")


class DedupeStep(PipelineStep[SearchStepContext]):
    span_name = "step.dedupe"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        span.set_attr("before_count", int(len(ctx.results or [])))
        kept, comparisons = self._dedupe(
            results=ctx.results,
            profile=ctx.profile,
        )
        ctx.results = kept
        span.set_attr("after_count", int(len(ctx.results or [])))
        span.set_attr("comparisons", int(comparisons))
        return ctx

    def _dedupe(
        self,
        *,
        results: list[ResultItem],
        profile: ProfileSettings | None,
    ) -> tuple[list[ResultItem], int]:
        profile = profile or self.settings.get_profile(self.settings.search.default_profile)
        results = self._dedupe_exact(results)

        title_tail_patterns = self._compile_patterns(profile.title_tail_patterns or [])
        domain_groups = profile.domain_groups or {}

        kept, comparisons = self._dedupe_fuzzy_lsh(
            results,
            threshold=float(profile.fuzzy_threshold),
            title_tail_patterns=title_tail_patterns,
            domain_groups=domain_groups,
        )
        return kept, comparisons

    def _dedupe_exact(self, results: list[ResultItem]) -> list[ResultItem]:
        seen: set[tuple[str, str]] = set()
        out: list[ResultItem] = []
        for r in results:
            title = clean_whitespace(r.title).lower()
            snippet = clean_whitespace(r.snippet).lower()
            fp = (title[:140], snippet[:260])
            if fp in seen:
                continue
            seen.add(fp)
            out.append(r)
        return out

    def _compile_patterns(self, patterns: list[str]) -> list[re.Pattern[str]]:
        out: list[re.Pattern[str]] = []
        for p in patterns:
            if not p:
                continue
            try:
                out.append(re.compile(p, re.IGNORECASE))
            except re.error:
                continue
        return out

    def _strip_title_tails(self, title: str, patterns: list[re.Pattern[str]]) -> str:
        cleaned = title or ""
        for pat in patterns:
            cleaned = pat.sub("", cleaned).strip()
        return cleaned

    def _fuzzy_key(
        self, r: ResultItem, title_tail_patterns: list[re.Pattern[str]]
    ) -> str:
        title = (r.title or "").strip()
        snippet = (r.snippet or "").strip()
        base = self._fuzzy_normalize(self._strip_title_tails(title, title_tail_patterns))
        if len(base) < 8 and snippet:
            base = f"{base} {self._fuzzy_normalize(snippet[:240])}".strip()
        return base

    def _fuzzy_normalize(self, text: str) -> str:
        lowered = (text or "").lower()
        lowered = re.sub(
            r"[^\w\u4e00-\u9fff\u3040-\u30ff]+", " ", lowered, flags=re.UNICODE
        )
        return _WS_RE.sub(" ", lowered).strip()

    def _canonical_site(self, domain: str, domain_groups: dict[str, list[str]]) -> str:
        d = (domain or "").lower()
        if not d:
            return "other"
        for group, needles in domain_groups.items():
            if any(n and n.lower() in d for n in (needles or [])):
                return group
        return d

    def _quality_score(self, r: ResultItem) -> int:
        return min(len(r.snippet or ""), 1200) + min(len(r.title or ""), 220)

    def _dedupe_fuzzy_lsh(
        self,
        results: list[ResultItem],
        *,
        threshold: float,
        title_tail_patterns: list[re.Pattern[str]],
        domain_groups: dict[str, list[str]],
    ) -> tuple[list[ResultItem], int]:
        if not results:
            return [], 0

        candidates = sorted(results, key=self._quality_score, reverse=True)

        keys = [self._fuzzy_key(r, title_tail_patterns) for r in candidates]
        sites = [self._canonical_site(r.domain, domain_groups) for r in candidates]
        hashes = [simhash64(keys[i]) for i in range(len(candidates))]

        buckets: list[dict[int, list[int]]] = [{} for _ in range(4)]
        for idx, h in enumerate(hashes):
            for band in range(4):
                key = int((h >> (band * 16)) & 0xFFFF)
                buckets[band].setdefault(key, []).append(idx)

        kept_idx: list[int] = []
        dropped: set[int] = set()
        comparisons = 0

        for i in range(len(candidates)):
            if i in dropped:
                continue
            kept_idx.append(i)

            neighbor_idxs: set[int] = set()
            h = hashes[i]
            for band in range(4):
                key = int((h >> (band * 16)) & 0xFFFF)
                for j in buckets[band].get(key, []):
                    if j <= i:
                        continue
                    neighbor_idxs.add(j)

            for j in neighbor_idxs:
                if j in dropped:
                    continue
                comparisons += 1
                th = threshold if sites[i] == sites[j] else min(0.94, threshold + 0.06)
                if hybrid_similarity(keys[i], keys[j]) >= th:
                    dropped.add(j)

        kept = [candidates[i] for i in kept_idx if i not in dropped]

        kept_id = {id(x) for x in kept}
        ordered = [r for r in results if id(r) in kept_id]
        return ordered, comparisons


__all__ = ["DedupeStep"]
