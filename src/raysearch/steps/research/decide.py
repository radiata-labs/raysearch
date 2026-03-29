from __future__ import annotations

import re
from typing import Any
from typing_extensions import override

from raysearch.components.llm.base import LLMClientBase
from raysearch.components.provider.base import SearchProviderBase
from raysearch.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from raysearch.dependencies import Depends
from raysearch.models.steps.research import (
    InformationGainEstimate,
    RoundStepContext,
)
from raysearch.models.steps.research.payloads import (
    ContentReviewPayload,
    ResearchDecideSignalPayload,
)
from raysearch.steps.base import StepBase
from raysearch.steps.research.prompt import build_decide_prompt_messages
from raysearch.steps.research.schema import build_decide_schema
from raysearch.steps.research.utils import (
    resolve_research_model,
    source_authority_score,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


class ResearchDecideStep(StepBase[RoundStepContext]):
    llm: LLMClientBase = Depends()
    provider: SearchProviderBase = Depends()

    # Thresholds for information-gain-based stopping
    _MIN_INFORMATION_GAIN_THRESHOLD = 0.15
    _LOW_CONFIDENCE_BOOST_THRESHOLD = 0.3
    _HIGH_UNRESOLVED_CONFLICTS_THRESHOLD = 2

    # Advanced thresholds for uncertainty-aware stopping
    _HIGH_EPISTEMIC_UNCERTAINTY_THRESHOLD = 0.5
    _LOW_COVERAGE_THRESHOLD = 0.4
    _MINIMUM_ROUNDS_FOR_CONFIDENT_STOP = 2

    @override
    async def run_inner(self, ctx: RoundStepContext) -> RoundStepContext:
        if ctx.run.current is None:
            return ctx
        budget = ctx.run.limits
        round_state = ctx.run.current
        if round_state.needs_resume:
            round_state.waiting_for_budget = True
            round_state.waiting_reason = (
                round_state.waiting_reason or "budget_resume_required"
            )
            return ctx
        unresolved_conflict_topics = self._get_unresolved_conflict_topics(
            content_review=round_state.content_review
        )
        missing_entities = list(round_state.missing_entities)
        critical_gaps = (
            list(round_state.content_review.remaining_gaps)
            if round_state.content_review is not None
            else []
        )

        gap_objectives = [f"gap:{item}" for item in critical_gaps]
        entity_objectives = [f"missing_entity:{item}" for item in missing_entities]
        conflict_objectives = [
            f"conflict:{item}" for item in unresolved_conflict_topics
        ]
        remaining_objectives = self._merge_preserving_order(
            [
                *gap_objectives,
                *entity_objectives,
                *conflict_objectives,
            ]
        )

        # Compute information gain estimate
        info_gain = self._estimate_information_gain(ctx=ctx)
        round_state.information_gain = info_gain

        llm_signal = await self._query_decide_signal(ctx=ctx)
        llm_prefers_continue = llm_signal.continue_research
        next_queries = list(llm_signal.next_queries[: budget.max_queries_per_round])
        query_objectives = [f"query:{item.query}" for item in next_queries]
        remaining_objectives = self._merge_preserving_order(
            [
                *remaining_objectives,
                *query_objectives,
            ]
        )
        search_exhausted = ctx.run.allocation.search_remaining <= 0
        fetch_exhausted = ctx.run.allocation.fetch_remaining <= 0
        can_search_now = (not search_exhausted) and (not fetch_exhausted)
        can_explore_without_search = self._can_continue_with_explore_only(ctx=ctx)
        min_rounds_per_track = max(1, ctx.run.limits.min_rounds_per_track)
        must_continue_for_min_rounds = (len(ctx.run.history) + 1) < min_rounds_per_track
        can_execute_next_round = (
            can_search_now and bool(next_queries)
        ) or can_explore_without_search

        # Enhanced stopping criteria using information gain
        should_stop_for_low_gain = (
            info_gain.estimated_gain < self._MIN_INFORMATION_GAIN_THRESHOLD
            and info_gain.marginal_gain < 0.3
            and not remaining_objectives
        )

        # Override stop decision if confidence is low but there's potential
        should_boost_for_low_confidence = (
            round_state.confidence < self._LOW_CONFIDENCE_BOOST_THRESHOLD
            and info_gain.confidence_improvement_potential > 0.4
        )

        # Override if there are too many unresolved conflicts
        should_boost_for_conflicts = (
            len(unresolved_conflict_topics) > self._HIGH_UNRESOLVED_CONFLICTS_THRESHOLD
        )

        # Advanced: Check uncertainty quantification
        uncertainty = round_state.uncertainty
        high_epistemic_uncertainty = (
            uncertainty.epistemic_uncertainty
            > self._HIGH_EPISTEMIC_UNCERTAINTY_THRESHOLD
        )

        # Advanced: Coverage-based stopping
        low_coverage = (
            round_state.coverage_ratio < self._LOW_COVERAGE_THRESHOLD
            and info_gain.coverage_gap_score > 0.3
        )

        # Advanced: Insufficient rounds for confident stop
        insufficient_rounds = (
            len(ctx.run.history) + 1 < self._MINIMUM_ROUNDS_FOR_CONFIDENT_STOP
        )

        # Compute adaptive stop readiness
        stop_ready = (
            (not llm_prefers_continue)
            and not remaining_objectives
            and not should_boost_for_low_confidence
            and not should_boost_for_conflicts
            and not high_epistemic_uncertainty
            and not low_coverage
        ) or should_stop_for_low_gain

        # Apply minimum rounds override
        if stop_ready and insufficient_rounds and can_execute_next_round:
            stop_ready = False
            ctx.run.notes.append("Minimum rounds requirement preventing early stop.")

        stop = False
        stop_reason = ""
        if fetch_exhausted:
            stop = True
            stop_reason = "max_fetch_calls"
        elif search_exhausted and not can_explore_without_search:
            stop = True
            stop_reason = "max_search_calls"
        elif (
            (
                must_continue_for_min_rounds
                and (can_search_now or can_explore_without_search)
            )
            or (remaining_objectives and can_execute_next_round)
            or (should_boost_for_low_confidence and can_execute_next_round)
            or (should_boost_for_conflicts and can_execute_next_round)
        ):
            stop = False
        elif stop_ready:
            stop = True
            stop_reason = "model_stop"
        elif not can_execute_next_round:
            stop = True
            stop_reason = "no_executable_path"

        round_state.stop = stop
        round_state.stop_reason = stop_reason
        ctx.run.next_queries = [] if stop else next_queries
        round_state.waiting_for_budget = False
        round_state.waiting_reason = ""
        ctx.run.archive_current_round()
        if llm_signal.reason:
            ctx.run.notes.append(f"Decide signal: {llm_signal.reason}")
        if info_gain.estimated_gain > 0:
            ctx.run.notes.append(
                f"Information gain estimate: {info_gain.estimated_gain:.2f} "
                f"(marginal: {info_gain.marginal_gain:.2f})"
            )
        await self.tracker.info(
            name="research.decide.summary",
            request_id=ctx.request_id,
            step="research.decide",
            data={
                "success": True,
                "next_queries": len(next_queries),
                "remaining_objectives_count": len(remaining_objectives),
                "stop": stop,
                "stop_reason": stop_reason or "n/a",
                "information_gain": info_gain.estimated_gain,
                "confidence": round_state.confidence,
                "coverage_ratio": round_state.coverage_ratio,
                "epistemic_uncertainty": uncertainty.epistemic_uncertainty,
                "unresolved_conflicts": len(unresolved_conflict_topics),
            },
        )
        await self.tracker.debug(
            name="research.decide.summary.detail",
            request_id=ctx.request_id,
            step="research.decide",
            data={
                "success": True,
                "llm_prefers_continue": llm_prefers_continue,
                "min_rounds_per_track": min_rounds_per_track,
                "must_continue_for_min_rounds": must_continue_for_min_rounds,
                "stop_ready": stop_ready,
                "can_search_now": can_search_now,
                "can_explore_without_search": can_explore_without_search,
                "can_execute_next_round": can_execute_next_round,
                "remaining_objectives": remaining_objectives,
                "next_queries": [item.model_dump(mode="json") for item in next_queries],
                "llm_reason": llm_signal.reason,
                "information_gain_detail": info_gain.model_dump(),
            },
        )
        if stop:
            ctx.run.stop = True
            ctx.run.stop_reason = stop_reason
        return ctx

    def _can_continue_with_explore_only(self, *, ctx: RoundStepContext) -> bool:
        if ctx.run.current is None:
            return False
        if ctx.run.allocation.fetch_remaining <= 0:
            return False
        return ctx.run.link_candidates_round == ctx.run.current.round_index and bool(
            ctx.run.link_candidates
        )

    async def _query_decide_signal(
        self,
        *,
        ctx: RoundStepContext,
    ) -> ResearchDecideSignalPayload:
        model = resolve_research_model(
            settings=self.settings,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        routes = resolve_engine_selection_routes(
            settings=self.settings,
            subsystem="research",
            provider=self.provider,
        )
        engine_selection_context = build_engine_selection_context(routes=routes)
        try:
            result = await self.llm.create(
                model=model,
                messages=build_decide_prompt_messages(
                    ctx=ctx,
                    engine_selection_context=engine_selection_context,
                ),
                response_format=ResearchDecideSignalPayload,
                format_override=build_decide_schema(
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
                    "prompt_tokens": int(result.usage.prompt_tokens),
                    "completion_tokens": int(result.usage.completion_tokens),
                    "total_tokens": int(result.usage.total_tokens),
                },
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="research.decide.failed",
                request_id=ctx.request_id,
                step="research.decide",
                error_code="research_decide_signal_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "model": str(model),
                },
            )
            raise
        return result.data

    def _get_unresolved_conflict_topics(
        self,
        *,
        content_review: ContentReviewPayload | None,
    ) -> list[str]:
        if content_review is None:
            return []
        return [
            r.topic
            for r in content_review.conflict_resolutions
            if r.status in {"unresolved", "insufficient"}
        ]

    def _merge_preserving_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for item in values:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    def _estimate_information_gain(
        self,
        *,
        ctx: RoundStepContext,
    ) -> InformationGainEstimate:
        """Estimate potential information gain from additional research rounds."""
        sources = ctx.knowledge.sources
        total_sources = len(sources)

        missing_entities = []
        if ctx.run.current and ctx.run.current.overview_review:
            missing_entities = list(ctx.run.current.overview_review.missing_entities)

        entity_gap = len(missing_entities) / max(1, len(ctx.task.entities) + 1)

        domains = set()
        for source in sources:
            domain = self._extract_domain_from_url(source.canonical_url or source.url)
            if domain:
                domains.add(domain)

        if domains:
            sources_per_domain = total_sources / len(domains)
            saturation = min(1.0, sources_per_domain / 5.0)
        else:
            saturation = 0.0

        novelty_potential = max(0.0, 1.0 - saturation)

        round_count = len(ctx.run.history)
        if round_count <= 1:
            marginal_gain = 0.8
        elif round_count <= 3:
            marginal_gain = 0.5
        else:
            marginal_gain = max(0.1, 0.3 * (0.7 ** (round_count - 3)))

        coverage_ratio = ctx.run.current.coverage_ratio if ctx.run.current else 0.0
        coverage_gap_score = max(0.0, 1.0 - coverage_ratio)

        current_confidence = 0.0
        if ctx.run.current:
            current_confidence = ctx.run.current.confidence
        confidence_potential = max(0.0, 1.0 - abs(current_confidence))

        semantic_saturation = self._compute_semantic_saturation(
            sources=sources,
            _subthemes=list(ctx.task.subthemes),
        )

        quality_trajectory = self._compute_quality_trajectory(ctx=ctx)

        conflict_potential = self._compute_conflict_resolution_potential(ctx=ctx)

        query_effectiveness = self._predict_query_effectiveness(ctx=ctx)

        base_gain = (
            0.25 * entity_gap
            + 0.20 * novelty_potential
            + 0.20 * coverage_gap_score
            + 0.15 * confidence_potential
            + 0.10 * (1.0 - semantic_saturation)
            + 0.10 * conflict_potential
        )

        adjusted_gain = base_gain * quality_trajectory * query_effectiveness
        estimated_gain = min(1.0, adjusted_gain * marginal_gain)

        return InformationGainEstimate(
            estimated_gain=estimated_gain,
            marginal_gain=marginal_gain,
            diminishing_returns_factor=1.0 - marginal_gain,
            coverage_gap_score=coverage_gap_score,
            novelty_potential=novelty_potential,
            confidence_improvement_potential=confidence_potential,
        )

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract the registered domain from a URL."""
        from urllib.parse import urlsplit

        from raysearch.utils import clean_whitespace

        try:
            parsed = urlsplit(url)
            host = clean_whitespace(parsed.netloc).casefold()
            host = host.removeprefix("www.")
            if ":" in host:
                host = host.split(":")[0]
            return host
        except Exception:  # noqa: S112
            return ""

    def _compute_semantic_saturation(
        self,
        *,
        sources: list[Any],
        _subthemes: list[str],
    ) -> float:
        if not sources or len(sources) < 3:
            return 0.0

        source_vocabularies: list[set[str]] = []
        for source in sources:
            text = (source.overview + " " + source.content).casefold()
            words = set(_TOKEN_PATTERN.findall(text))
            words = {w for w in words if len(w) > 3}
            source_vocabularies.append(words)

        if not source_vocabularies:
            return 0.0

        similarities: list[float] = []
        for i, vocab_a in enumerate(source_vocabularies):
            for vocab_b in source_vocabularies[i + 1 :]:
                if not vocab_a or not vocab_b:
                    continue
                intersection = len(vocab_a & vocab_b)
                union = len(vocab_a | vocab_b)
                if union > 0:
                    similarities.append(intersection / union)

        if not similarities:
            return 0.0

        avg_similarity = sum(similarities) / len(similarities)
        return min(1.0, avg_similarity * 1.5)

    def _compute_quality_trajectory(self, *, ctx: RoundStepContext) -> float:
        from raysearch.models.steps.research import ResearchSource

        history = ctx.run.history
        if len(history) < 2:
            return 1.0

        round_authorities: list[float] = []
        sources_by_round: dict[int, list[ResearchSource]] = {}

        for source in ctx.knowledge.sources:
            round_idx = source.round_index
            if round_idx not in sources_by_round:
                sources_by_round[round_idx] = []
            sources_by_round[round_idx].append(source)

        for round_idx in sorted(sources_by_round.keys()):
            round_sources = sources_by_round[round_idx]
            if round_sources:
                avg_authority = sum(
                    source_authority_score(s) for s in round_sources
                ) / len(round_sources)
                round_authorities.append(avg_authority)

        if len(round_authorities) < 2:
            return 1.0

        recent_avg = sum(round_authorities[-2:]) / 2
        earlier_avg = sum(round_authorities[:-2]) / max(1, len(round_authorities) - 2)

        if earlier_avg > 0:
            trend = recent_avg / earlier_avg
            return min(1.0, max(0.5, trend))

        return 1.0

    def _compute_conflict_resolution_potential(self, *, ctx: RoundStepContext) -> float:
        current = ctx.run.current
        if not current:
            return 0.0

        unresolved_count = 0
        total_conflicts = 0

        if current.content_review:
            total_conflicts = len(current.content_review.conflict_resolutions)
            unresolved_count = sum(
                1
                for r in current.content_review.conflict_resolutions
                if r.status in {"unresolved", "insufficient"}
            )
        elif current.overview_review:
            total_conflicts = len(current.overview_review.conflict_topics)
            unresolved_count = total_conflicts

        if total_conflicts == 0:
            return 1.0

        resolution_potential = unresolved_count / max(1, total_conflicts)
        return min(1.0, resolution_potential * 0.8 + 0.2)

    def _predict_query_effectiveness(self, *, ctx: RoundStepContext) -> float:
        history = ctx.run.history
        if not history:
            return 1.0

        total_queries = 0
        total_sources_from_queries = 0

        for round_state in history:
            query_count = len(round_state.queries)
            if query_count > 0:
                total_queries += query_count
                round_sources = sum(
                    1
                    for s in ctx.knowledge.sources
                    if s.round_index == round_state.round_index
                )
                total_sources_from_queries += round_sources

        if total_queries == 0:
            return 1.0

        sources_per_query = total_sources_from_queries / total_queries

        effectiveness = min(1.0, sources_per_query / 2.0)

        if sources_per_query > 3:
            effectiveness = max(0.7, effectiveness * 0.9)

        return effectiveness


__all__ = ["ResearchDecideStep"]
