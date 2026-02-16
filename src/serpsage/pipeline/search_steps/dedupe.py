from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import SearchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.utils import clean_whitespace, normalize_text

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime

_WS_RE = re.compile(r"\s+")


def char_ngrams(text: str, n: int) -> set[str]:
    compact = (text or "").replace(" ", "")
    if not compact:
        return set()
    if len(compact) < n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def hybrid_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    seq = SequenceMatcher(None, a, b).ratio()
    jac = jaccard(char_ngrams(a, 2), char_ngrams(b, 2))
    return float(max(seq * 0.95, jac))


def simhash64(text: str) -> int:
    """A tiny 64-bit simhash for near-duplicate detection."""
    t = normalize_text(text)
    if not t:
        return 0
    feats = [f for f in t.split(" ") if f]
    if not feats:
        return 0

    vec = [0] * 64
    for f in feats:
        h = int.from_bytes(
            hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for i in range(64):
            bit = 1 if (h >> i) & 1 else -1
            vec[i] += bit

    out = 0
    for i in range(64):
        if vec[i] >= 0:
            out |= 1 << i
    return out


class DedupeStep(PipelineStep[SearchStepContext]):
    span_name = "step.dedupe"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        span.set_attr("before_count", int(len(ctx.results or [])))
        kept, comparisons = self._dedupe(results=ctx.results)
        ctx.results = kept
        span.set_attr("after_count", int(len(ctx.results or [])))
        span.set_attr("comparisons", int(comparisons))
        return ctx

    def _dedupe(
        self,
        *,
        results: list[ResultItem],
    ) -> tuple[list[ResultItem], int]:
        results = self._dedupe_exact(results)

        kept, comparisons = self._dedupe_fuzzy_lsh(
            results,
            threshold=float(self.settings.search.fuzzy_threshold),
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

    def _fuzzy_key(self, r: ResultItem) -> str:
        title = (r.title or "").strip()
        snippet = (r.snippet or "").strip()
        base = self._fuzzy_normalize(title)
        if len(base) < 8 and snippet:
            base = f"{base} {self._fuzzy_normalize(snippet[:240])}".strip()
        return base

    def _fuzzy_normalize(self, text: str) -> str:
        lowered = (text or "").lower()
        lowered = re.sub(
            r"[^\w\u4e00-\u9fff\u3040-\u30ff]+", " ", lowered, flags=re.UNICODE
        )
        return _WS_RE.sub(" ", lowered).strip()

    def _canonical_site(self, domain: str) -> str:
        d = (domain or "").lower()
        return d or "other"

    def _rank_score(self, r: ResultItem) -> int:
        return min(len(r.snippet or ""), 1200) + min(len(r.title or ""), 220)

    def _dedupe_fuzzy_lsh(
        self,
        results: list[ResultItem],
        *,
        threshold: float,
    ) -> tuple[list[ResultItem], int]:
        if not results:
            return [], 0

        candidates = sorted(results, key=self._rank_score, reverse=True)

        keys = [self._fuzzy_key(r) for r in candidates]
        sites = [self._canonical_site(r.domain) for r in candidates]
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
