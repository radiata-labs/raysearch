from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Literal
from typing_extensions import override
from urllib.parse import urlsplit

from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from serpsage.components.rank.base import RankerBase
from serpsage.dependencies import Depends
from serpsage.models.steps.research import (
    CorroborationScore,
    ResearchSource,
    RoundStepContext,
    SourceDiversityMetrics,
)
from serpsage.models.steps.research.payloads import OverviewReviewPayload
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_overview_prompt_messages
from serpsage.steps.research.schema import build_overview_schema
from serpsage.steps.research.utils import (
    canonicalize_url,
    pick_sources_by_ids,
    rerank_research_sources,
    resolve_research_model,
    source_authority_score,
)
from serpsage.utils import clean_whitespace

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


class ResearchOverviewStep(StepBase[RoundStepContext]):
    _CONTEXT_NEW_RESULT_TARGET_RATIO = 0.60
    _CONTEXT_MIN_HISTORY_SOURCES = 3

    llm: LLMClientBase = Depends()
    ranker: RankerBase = Depends()
    provider: SearchProviderBase = Depends()

    @override
    async def run_inner(self, ctx: RoundStepContext) -> RoundStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.run.stop or ctx.run.current is None:
            return ctx
        if not ctx.run.current.is_review_ready:
            return ctx
        all_sources = list(ctx.knowledge.sources)
        if not all_sources:
            ctx.run.current.overview_review = self._empty_review()
            ctx.run.current.need_content_source_ids = []
            ctx.run.current.context_source_ids = []
            return ctx
        mode_depth = ctx.run.limits
        review_source_window = max(1, mode_depth.review_source_window)
        new_result_target_ratio, min_history_sources = (
            self._resolve_context_mix_targets(
                ctx=ctx,
                sources=all_sources,
            )
        )
        context_source_ids = self._select_context_source_ids(
            ctx=ctx,
            round_index=ctx.run.current.round_index,
            topk=review_source_window,
            new_result_target_ratio=new_result_target_ratio,
            min_history_sources=min_history_sources,
        )
        ctx.run.current.context_source_ids = list(context_source_ids)
        sources = pick_sources_by_ids(
            sources=all_sources,
            source_ids=context_source_ids,
        )
        if not sources:
            ctx.run.current.overview_review = self._empty_review()
            ctx.run.current.need_content_source_ids = []
            return ctx
        sources = await rerank_research_sources(
            ctx=ctx,
            ranker=self.ranker,
            sources=sources,
            query=ctx.task.question,
        )
        ctx.run.current.context_source_ids = [item.source_id for item in sources]
        model = resolve_research_model(
            settings=self.settings,
            stage="overview",
            fallback=self.settings.answer.generate.use_model,
        )
        routes = resolve_engine_selection_routes(
            settings=self.settings,
            subsystem="research",
            provider=self.provider,
        )
        engine_selection_context = build_engine_selection_context(routes=routes)
        try:
            chat_result = await self.llm.create(
                model=model,
                messages=build_overview_prompt_messages(
                    ctx=ctx,
                    sources=sources,
                    now_utc=now_utc,
                    engine_selection_context=engine_selection_context,
                ),
                response_format=OverviewReviewPayload,
                format_override=build_overview_schema(
                    max_queries=ctx.run.limits.max_queries_per_round,
                    select_engines=bool(routes),
                ),
                retries=self.settings.research.llm_self_heal_retries,
            )
            await self.meter.record(
                name="llm.tokens",
                request_id=ctx.request_id,
                model=str(model),
                unit="token",
                tokens={
                    "prompt_tokens": int(chat_result.usage.prompt_tokens),
                    "completion_tokens": int(chat_result.usage.completion_tokens),
                    "total_tokens": int(chat_result.usage.total_tokens),
                },
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="research.overview.failed",
                request_id=ctx.request_id,
                step="research.overview",
                error_code="research_overview_review_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "round_index": ctx.run.current.round_index,
                },
            )
            raise
        payload = chat_result.data
        ctx.run.current.overview_review = payload
        ctx.run.current.need_content_source_ids = self._resolve_need_content_source_ids(
            payload=payload,
        )

        # Compute source diversity for this round
        ctx.run.current.source_diversity = self._compute_source_diversity(sources)

        # Compute corroboration scores for key findings
        ctx.run.current.corroborations = self._compute_corroboration_scores(
            findings=payload.findings,
            sources=sources,
        )

        if payload.findings:
            ctx.run.notes.extend(payload.findings[:3])
            ctx.run.current.overview_summary = " | ".join(payload.findings[:3])
        if payload.covered_subthemes:
            ctx.knowledge.covered_subthemes = self._merge_preserving_order(
                left=ctx.knowledge.covered_subthemes,
                right=list(payload.covered_subthemes),
            )
        total = max(1, len(ctx.task.subthemes))
        ctx.run.current.coverage_ratio = min(
            1.0,
            len(ctx.knowledge.covered_subthemes) / total,
        )
        await self.tracker.info(
            name="research.style.applied",
            request_id=ctx.request_id,
            step="research.overview",
            data={
                "success": True,
                "sources_reviewed": len(sources),
                "findings": len(payload.findings),
                "confidence": payload.confidence,
                "coverage_ratio": ctx.run.current.coverage_ratio,
                "source_diversity": ctx.run.current.source_diversity.perspective_diversity_score,
            },
        )
        await self.tracker.debug(
            name="research.style.applied.detail",
            request_id=ctx.request_id,
            step="research.overview",
            data={
                "success": True,
                "report_style_selected": ctx.task.style,
                "style_applied_stage": "overview",
                "mode_depth_profile": mode_depth.mode_key,
                "review_source_window_effective": review_source_window,
                "overview_new_result_target_ratio": new_result_target_ratio,
                "overview_min_history_sources": min_history_sources,
                "context_source_ids": ctx.run.current.context_source_ids,
                "need_content_source_ids": ctx.run.current.need_content_source_ids,
                "covered_subthemes": len(payload.covered_subthemes),
                "missing_entities": payload.missing_entities,
                "conflict_topics": len(payload.conflict_topics),
                "corroboration_count": len(ctx.run.current.corroborations),
            },
        )
        return ctx

    def _compute_corroboration_scores(
        self,
        *,
        findings: list[str],
        sources: list[ResearchSource],
    ) -> list[CorroborationScore]:
        """Compute cross-source corroboration scores for findings.

        Uses semantic similarity analysis to identify supporting and
        contradicting evidence across sources.
        """
        if not findings or not sources:
            return []

        corroborations: list[CorroborationScore] = []

        for finding in findings[:8]:  # Limit to top findings
            # Use semantic analysis for more accurate corroboration
            supporting_ids: list[int] = []
            contradicting_ids: list[int] = []
            key_evidence_points: list[str] = []

            for source in sources:
                content = (source.overview + " " + source.content).casefold()

                # Compute semantic similarity
                similarity = self._compute_semantic_similarity(finding, content)

                # Check for supporting evidence
                if similarity > 0.25:
                    # Check if this is supporting or contradicting
                    evidence_strength = self._compute_evidence_strength(
                        [source], finding
                    )

                    if evidence_strength > 0.1:
                        supporting_ids.append(source.source_id)
                        # Extract key evidence snippet
                        if source.overview:
                            key_evidence_points.append(
                                source.overview[:150].rsplit(".", 1)[-1].strip() + "..."
                            )
                    elif evidence_strength < -0.1:
                        contradicting_ids.append(source.source_id)

            if supporting_ids or contradicting_ids:
                total = len(supporting_ids) + len(contradicting_ids)
                ratio = len(supporting_ids) / max(1, total)

                consensus: Literal["strong", "moderate", "weak", "conflicted"]
                if ratio >= 0.75 and len(supporting_ids) >= 3:
                    consensus = "strong"
                elif ratio >= 0.5 and len(supporting_ids) >= 2:
                    consensus = "moderate"
                elif ratio >= 0.25:
                    consensus = "weak"
                else:
                    consensus = "conflicted"

                corroborations.append(
                    CorroborationScore(
                        claim=finding[:200],
                        supporting_source_ids=supporting_ids[:8],
                        contradicting_source_ids=contradicting_ids[:8],
                        corroboration_ratio=ratio,
                        consensus_strength=consensus,
                        key_evidence_points=key_evidence_points[:5],
                    )
                )

        return corroborations

    def _empty_review(self) -> OverviewReviewPayload:
        return OverviewReviewPayload(
            findings=[],
            conflict_topics=[],
            covered_subthemes=[],
            need_content_source_ids=[],
            missing_entities=[],
            confidence=0.0,
        )

    def _resolve_need_content_source_ids(
        self,
        *,
        payload: OverviewReviewPayload,
    ) -> list[int]:
        if not payload.need_content_source_ids:
            return []
        out: list[int] = []
        seen: set[int] = set()
        for source_id in payload.need_content_source_ids:
            if source_id in seen:
                continue
            seen.add(source_id)
            out.append(source_id)
        return out

    def _merge_preserving_order(
        self,
        *,
        left: list[str],
        right: list[str],
    ) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for item in [*left, *right]:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    def _resolve_context_mix_targets(
        self,
        *,
        ctx: RoundStepContext,
        sources: list[ResearchSource],
    ) -> tuple[float, int]:
        mode_key = ctx.run.limits.mode_key
        if mode_key == "research-fast":
            base_ratio = 0.70
            base_min_history = 1
        elif mode_key == "research-pro":
            base_ratio = 0.55
            base_min_history = 3
        else:
            base_ratio = self._CONTEXT_NEW_RESULT_TARGET_RATIO
            base_min_history = 2
        round_index = ctx.run.current.round_index if ctx.run.current else 0
        new_count = sum(
            1 for item in sources if int(getattr(item, "round_index", 0)) == round_index
        )
        total_count = len(sources)
        history_count = max(0, total_count - new_count)
        if new_count <= 0:
            return 0.0, min(base_min_history, max(1, history_count))
        if history_count <= 0:
            return 1.0, 0
        ratio = base_ratio
        if round_index <= 1:
            ratio = max(ratio, 0.70)
        if new_count <= 2:
            ratio = min(0.85, ratio + 0.15)
        if history_count <= 2:
            ratio = max(0.35, ratio - 0.20)
        min_history = min(max(1, history_count), base_min_history)
        if round_index <= 1:
            min_history = min(min_history, 1)
        return max(0.0, min(1.0, ratio)), max(0, min_history)

    def _compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between two texts using vocabulary overlap.

        Returns a score from 0.0 (no similarity) to 1.0 (identical vocabulary).
        Uses Jaccard similarity on token sets.
        """
        tokens_a = set(_TOKEN_PATTERN.findall(text_a.casefold()))
        tokens_b = set(_TOKEN_PATTERN.findall(text_b.casefold()))

        tokens_a = {t for t in tokens_a if len(t) > 2}
        tokens_b = {t for t in tokens_b if len(t) > 2}

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)

        return intersection / union if union > 0 else 0.0

    def _compute_evidence_strength(
        self, sources: list[ResearchSource], claim: str
    ) -> float:
        """Compute evidence strength for a claim based on source support.

        Higher scores indicate stronger evidential support.
        """
        if not sources or not claim:
            return 0.0

        claim_lower = claim.casefold()
        claim_tokens = set(_TOKEN_PATTERN.findall(claim_lower))
        claim_tokens = {t for t in claim_tokens if len(t) > 2}

        if not claim_tokens:
            return 0.0

        supporting_score = 0.0
        contradicting_score = 0.0

        contradiction_indicators = [
            "however",
            "but",
            "contrary",
            "dispute",
            "challenge",
            "refute",
            "debunk",
            "incorrect",
            "false",
            "misleading",
        ]

        for source in sources:
            text = (source.overview + " " + source.content).casefold()
            text_tokens = set(_TOKEN_PATTERN.findall(text))
            text_tokens = {t for t in text_tokens if len(t) > 2}

            overlap = claim_tokens & text_tokens
            overlap_ratio = len(overlap) / len(claim_tokens) if claim_tokens else 0

            if overlap_ratio > 0.3:
                authority = source_authority_score(source)
                base_score = overlap_ratio * authority

                has_contradiction = any(
                    indicator in text for indicator in contradiction_indicators
                )

                if has_contradiction:
                    contradicting_score += base_score * 0.5
                else:
                    supporting_score += base_score

        total = supporting_score + contradicting_score
        if total == 0:
            return 0.0

        return min(1.0, max(-0.5, (supporting_score - contradicting_score) / total))

    def _compute_source_diversity(
        self, sources: list[ResearchSource]
    ) -> SourceDiversityMetrics:
        """Compute diversity metrics for a set of sources."""
        if not sources:
            return SourceDiversityMetrics()

        domain_counts: dict[str, int] = {}
        content_type_counts: dict[str, int] = {}
        authority_tier_counts: dict[str, int] = {}

        for source in sources:
            domain = self._extract_domain_from_url(source.canonical_url or source.url)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            content_type = self._classify_content_type(source)
            content_type_counts[content_type] = (
                content_type_counts.get(content_type, 0) + 1
            )

            auth_score = source_authority_score(source)
            if auth_score >= 0.85:
                tier = "high"
            elif auth_score >= 0.60:
                tier = "medium"
            else:
                tier = "low"
            authority_tier_counts[tier] = authority_tier_counts.get(tier, 0) + 1

        total_sources = len(sources)
        unique_domains = len(domain_counts)
        if unique_domains <= 1:
            perspective_diversity = 0.0
        else:
            entropy = 0.0
            for count in domain_counts.values():
                p = count / total_sources
                entropy -= p * math.log2(p)
            max_entropy = math.log2(unique_domains)
            perspective_diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        geographic_diversity = 0.5

        return SourceDiversityMetrics(
            unique_domains=unique_domains,
            domain_distribution=domain_counts,
            perspective_diversity_score=min(1.0, perspective_diversity),
            geographic_diversity_score=geographic_diversity,
            content_type_distribution=content_type_counts,
            authority_tier_distribution=authority_tier_counts,
        )

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract the registered domain from a URL."""
        try:
            parsed = urlsplit(url)
            host = clean_whitespace(parsed.netloc).casefold()
            host = host.removeprefix("www.")
            if ":" in host:
                host = host.split(":")[0]
            return host
        except Exception:  # noqa: S112
            return ""

    def _classify_content_type(self, source: ResearchSource) -> str:
        """Classify the content type of a source based on URL and title."""
        url = (source.canonical_url or source.url).casefold()
        title = source.title.casefold()
        path = ""
        try:
            parsed = urlsplit(url)
            path = parsed.path.casefold()
        except Exception:  # noqa: S112
            pass

        if any(
            hint in path or hint in title
            for hint in [
                "/docs",
                "/documentation",
                "/api",
                "documentation",
                "api reference",
            ]
        ):
            return "documentation"

        if any(
            hint in url or hint in title
            for hint in [
                "arxiv.org",
                "doi.org",
                "paper",
                "preprint",
                "whitepaper",
                "research",
            ]
        ):
            return "paper"

        if any(
            hint in url or hint in title
            for hint in [".gov", "official", "release", "announcement"]
        ):
            return "official"

        if any(
            hint in url or hint in title
            for hint in ["news", "article", "press", "report"]
        ):
            return "news"

        if any(
            hint in url or hint in title
            for hint in ["wiki", "encyclopedia", "reference", "guide"]
        ):
            return "reference"

        if any(
            hint in url
            for hint in ["blog", "medium.com", "substack.com", "wordpress.com"]
        ):
            return "blog"

        if any(
            hint in url
            for hint in [
                "forum",
                "discuss",
                "reddit.com",
                "stack overflow",
                "stackexchange",
            ]
        ):
            return "forum"

        return "article"

    def _select_context_source_ids(
        self,
        *,
        ctx: RoundStepContext,
        round_index: int,
        topk: int,
        new_result_target_ratio: float,
        min_history_sources: int,
    ) -> list[int]:
        """Select context source IDs for review."""
        limit = max(1, topk)
        ranked_ids = self._resolve_ranked_source_ids(ctx=ctx)
        if not ranked_ids:
            return []
        source_by_id = {source.source_id: source for source in ctx.knowledge.sources}
        new_ids = [
            source_id
            for source_id in ranked_ids
            if source_by_id[source_id].round_index == round_index
        ]
        history_ids = [
            source_id
            for source_id in ranked_ids
            if source_by_id[source_id].round_index != round_index
        ]
        history_by_authority = sorted(
            history_ids,
            key=lambda sid: (
                source_authority_score(source_by_id[sid]),
                float(ctx.knowledge.source_scores.get(sid, 0.0)),
                source_by_id[sid].round_index,
                sid,
            ),
            reverse=True,
        )
        target_new = min(
            len(new_ids),
            int(math.ceil(limit * float(max(0.0, min(1.0, new_result_target_ratio))))),
        )
        selected: list[int] = []
        selected.extend(new_ids[:target_new])
        selected.extend(history_by_authority[: max(0, limit - len(selected))])
        if len(selected) < limit:
            for source_id in new_ids[target_new:]:
                if source_id in selected:
                    continue
                selected.append(source_id)
                if len(selected) >= limit:
                    break
        if len(selected) < limit:
            for source_id in history_by_authority:
                if source_id in selected:
                    continue
                selected.append(source_id)
                if len(selected) >= limit:
                    break
        min_history = max(0, min_history_sources)
        history_needed = min(min_history, len(history_ids))
        history_selected = sum(1 for source_id in selected if source_id in history_ids)
        if history_selected < history_needed:
            for source_id in history_by_authority:
                if source_id in selected:
                    continue
                selected.append(source_id)
                history_selected += 1
                if history_selected >= history_needed:
                    break
        if len(selected) > limit:
            selected = self._trim_to_limit(
                selected=selected,
                source_by_id=source_by_id,
                round_index=round_index,
                limit=limit,
            )
        rank_index = {source_id: idx for idx, source_id in enumerate(ranked_ids)}
        deduped: list[int] = []
        seen: set[int] = set()
        for source_id in selected:
            if source_id in seen:
                continue
            seen.add(source_id)
            deduped.append(source_id)
        deduped.sort(key=lambda sid: rank_index.get(sid, 10**9))
        return deduped[:limit]

    def _resolve_ranked_source_ids(self, *, ctx: RoundStepContext) -> list[int]:
        """Resolve ranked source IDs from context."""
        source_ids: list[int] = []
        source_by_id = {source.source_id: source for source in ctx.knowledge.sources}
        seen_canonical: set[str] = set()
        for source_id in ctx.knowledge.ranked_source_ids:
            source = source_by_id.get(source_id)
            if source is None:
                continue
            canonical = source.canonical_url or canonicalize_url(source.url)
            if not canonical or canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            source_ids.append(source_id)
        if source_ids:
            return source_ids
        fallback = sorted(
            ctx.knowledge.sources,
            key=lambda item: (item.round_index, item.source_id),
            reverse=True,
        )
        out: list[int] = []
        for source in fallback:
            canonical = source.canonical_url or canonicalize_url(source.url)
            if not canonical or canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            out.append(source.source_id)
        return out

    def _trim_to_limit(
        self,
        *,
        selected: list[int],
        source_by_id: dict[int, ResearchSource],
        round_index: int,
        limit: int,
    ) -> list[int]:
        """Trim selected source IDs to limit."""
        out = list(selected)
        while len(out) > limit:
            removed = False
            for idx in range(len(out) - 1, -1, -1):
                source_id = out[idx]
                if source_by_id[source_id].round_index == round_index:
                    out.pop(idx)
                    removed = True
                    break
            if not removed:
                out.pop()
        return out


__all__ = ["ResearchOverviewStep"]
