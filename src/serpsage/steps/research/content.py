from __future__ import annotations

from datetime import UTC, datetime
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from serpsage.components.rank.base import RankerBase
from serpsage.dependencies import Depends
from serpsage.models.steps.research import (
    ReasoningChain,
    ReasoningStep,
    ResearchSource,
    RoundStepContext,
    UncertaintyQuantification,
)
from serpsage.models.steps.research.payloads import ContentReviewPayload
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_content_prompt_messages
from serpsage.steps.research.schema import build_content_schema
from serpsage.steps.research.utils import (
    pick_sources_by_ids,
    rerank_research_sources,
    resolve_research_model,
    source_authority_score,
)


class ResearchContentStep(StepBase[RoundStepContext]):
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
        review_source_window = max(1, ctx.run.limits.review_source_window)
        source_ids = list(
            ctx.run.current.need_content_source_ids
            or ctx.run.current.context_source_ids
        )
        source_ids = self._sort_source_ids_by_score(
            ctx=ctx,
            source_ids=source_ids,
        )[:review_source_window]
        if not source_ids:
            ctx.run.current.content_review = self._empty_review()
            return ctx
        selected_sources = pick_sources_by_ids(
            sources=ctx.knowledge.sources,
            source_ids=source_ids,
        )
        if not selected_sources:
            ctx.run.current.content_review = self._empty_review()
            return ctx
        selected_sources = await rerank_research_sources(
            ctx=ctx,
            ranker=self.ranker,
            sources=selected_sources,
            query=ctx.task.question,
        )
        source_ids = [item.source_id for item in selected_sources]
        model = resolve_research_model(
            settings=self.settings,
            stage="content",
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
                messages=build_content_prompt_messages(
                    ctx=ctx,
                    selected_sources=selected_sources,
                    source_ids=source_ids,
                    now_utc=now_utc,
                    engine_selection_context=engine_selection_context,
                ),
                response_format=ContentReviewPayload,
                format_override=build_content_schema(
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
                name="research.content.failed",
                request_id=ctx.request_id,
                step="research.content",
                error_code="research_content_review_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "round_index": ctx.run.current.round_index,
                },
            )
            raise
        payload = chat_result.data
        ctx.run.current.content_review = payload
        if payload.resolved_findings:
            ctx.run.current.content_summary = " | ".join(payload.resolved_findings[:3])
            ctx.run.notes.extend(payload.resolved_findings[:3])

        # Calculate final confidence from overview + content adjustment
        base_confidence = (
            ctx.run.current.overview_review.confidence
            if ctx.run.current.overview_review
            else 0.0
        )
        final_confidence = max(
            -1.0, min(1.0, base_confidence + payload.confidence_adjustment)
        )

        # Compute uncertainty quantification
        ctx.run.current.uncertainty = self._compute_uncertainty(
            content_review=payload,
            sources=selected_sources,
            final_confidence=final_confidence,
        )

        # Build reasoning chains for multi-hop reasoning support
        ctx.run.current.reasoning_chains = self._build_reasoning_chains(
            findings=payload.resolved_findings,
            sources=selected_sources,
            source_ids=source_ids,
        )

        await self.tracker.info(
            name="research.style.applied",
            request_id=ctx.request_id,
            step="research.content",
            data={
                "success": True,
                "sources_reviewed": len(selected_sources),
                "resolved_findings": len(payload.resolved_findings),
                "remaining_gaps": len(payload.remaining_gaps),
                "confidence": final_confidence,
                "epistemic_uncertainty": ctx.run.current.uncertainty.epistemic_uncertainty,
            },
        )
        await self.tracker.debug(
            name="research.style.applied.detail",
            request_id=ctx.request_id,
            step="research.content",
            data={
                "success": True,
                "report_style_selected": ctx.task.style,
                "style_applied_stage": "content",
                "mode_depth_profile": ctx.run.limits.mode_key,
                "review_source_window_effective": review_source_window,
                "source_ids": source_ids,
                "resolved_findings": payload.resolved_findings,
                "remaining_gaps": payload.remaining_gaps,
                "conflict_resolutions": [
                    {"topic": r.topic, "status": r.status}
                    for r in payload.conflict_resolutions
                ],
                "confidence_adjustment": payload.confidence_adjustment,
                "uncertainty_quantification": ctx.run.current.uncertainty.model_dump(),
            },
        )
        return ctx

    def _compute_uncertainty(
        self,
        *,
        content_review: ContentReviewPayload,
        sources: list[ResearchSource],
        final_confidence: float,
    ) -> UncertaintyQuantification:
        """Compute uncertainty quantification based on content review.

        Uses multi-factor analysis to separate epistemic (knowledge-based)
        uncertainty from aleatoric (inherent randomness) uncertainty.
        """
        # Epistemic uncertainty: based on knowledge gaps and unresolved conflicts
        unresolved_count = sum(
            1
            for r in content_review.conflict_resolutions
            if r.status in {"unresolved", "insufficient"}
        )
        gap_count = len(content_review.remaining_gaps)

        # More gaps and conflicts = higher epistemic uncertainty
        epistemic = min(1.0, (unresolved_count * 0.2 + gap_count * 0.1))

        # Aleatoric uncertainty: inherent randomness in the domain
        # Based on conflicting evidence even with good sources
        total_resolutions = len(content_review.conflict_resolutions)
        if total_resolutions > 0:
            aleatoric = unresolved_count / total_resolutions * 0.5
        else:
            aleatoric = 0.0

        # Adjust epistemic based on evidence quality and quantity
        if sources:
            avg_authority = sum(
                self._compute_source_authority(s) for s in sources
            ) / len(sources)
            # High authority sources reduce epistemic uncertainty
            epistemic *= max(0.3, 1.0 - avg_authority * 0.5)

        # Confidence interval based on final confidence and uncertainty
        uncertainty_adjustment = max(epistemic, aleatoric) * 0.5
        ci_lower = max(-1.0, final_confidence - uncertainty_adjustment)
        ci_upper = min(1.0, final_confidence + uncertainty_adjustment)

        # Identify uncertainty sources
        sources_list: list[str] = []
        if unresolved_count > 0:
            sources_list.append(f"unresolved_conflicts:{unresolved_count}")
        if gap_count > 0:
            sources_list.append(f"knowledge_gaps:{gap_count}")
        if final_confidence < 0.3:
            sources_list.append("low_confidence")
        if len(sources) < 3:
            sources_list.append("limited_evidence")

        # Add advanced uncertainty indicators
        if epistemic > 0.5:
            sources_list.append("high_epistemic_uncertainty")
        if aleatoric > 0.3:
            sources_list.append("domain_inherent_uncertainty")

        # Check for contradictory evidence patterns
        if content_review.conflict_resolutions:
            contradiction_ratio = unresolved_count / max(1, total_resolutions)
            if contradiction_ratio > 0.5:
                sources_list.append("high_contradiction_ratio")

        return UncertaintyQuantification(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            uncertainty_sources=sources_list,
            mitigating_evidence_count=len(content_review.resolved_findings),
        )

    def _compute_source_authority(self, source: ResearchSource) -> float:
        """Compute authority score for a source."""
        return source_authority_score(source)

    def _sort_source_ids_by_score(
        self,
        *,
        ctx: RoundStepContext,
        source_ids: list[int],
    ) -> list[int]:
        """Sort source IDs by their scores."""
        source_by_id = {source.source_id: source for source in ctx.knowledge.sources}
        out: list[int] = []
        seen: set[int] = set()
        for raw in source_ids:
            source_id = raw
            if source_id in seen or source_id not in source_by_id:
                continue
            seen.add(source_id)
            out.append(source_id)
        out.sort(
            key=lambda sid: (
                float(ctx.knowledge.source_scores.get(sid, 0.0)),
                source_authority_score(source_by_id[sid]),
                source_by_id[sid].round_index,
                sid,
            ),
            reverse=True,
        )
        return out

    def _build_reasoning_chains(
        self,
        *,
        findings: list[str],
        sources: list[ResearchSource],
        source_ids: list[int],
    ) -> list[ReasoningChain]:
        """Build reasoning chains from findings for multi-hop reasoning support.

        This method constructs simple reasoning chains that can be used
        to understand the logical flow from evidence to conclusions.
        """
        if not findings or not sources:
            return []

        chains: list[ReasoningChain] = []

        for idx, finding in enumerate(findings[:5]):
            # Create a simple reasoning chain for each finding
            steps: list[ReasoningStep] = []

            # Step 1: Premise (evidence observation)
            relevant_sources = [
                s
                for s in sources
                if finding.casefold()[:50] in (s.overview + s.content).casefold()
            ][:3]

            if relevant_sources:
                steps.append(
                    ReasoningStep(
                        step_id=f"chain_{idx}_premise",
                        step_type="premise",
                        claim=f"Evidence observed from {len(relevant_sources)} source(s)",
                        evidence_source_ids=[s.source_id for s in relevant_sources],
                        confidence=0.8 if len(relevant_sources) >= 2 else 0.6,
                        reasoning_logic="Direct observation from source content",
                        dependencies=[],
                    )
                )

            # Step 2: Inference
            steps.append(
                ReasoningStep(
                    step_id=f"chain_{idx}_inference",
                    step_type="inference",
                    claim=finding[:200],
                    evidence_source_ids=[s.source_id for s in relevant_sources]
                    if relevant_sources
                    else [],
                    confidence=0.7,
                    reasoning_logic="Synthesized from evidence analysis",
                    dependencies=[f"chain_{idx}_premise"] if relevant_sources else [],
                )
            )

            # Step 3: Conclusion
            steps.append(
                ReasoningStep(
                    step_id=f"chain_{idx}_conclusion",
                    step_type="conclusion",
                    claim=finding[:150],
                    evidence_source_ids=[],
                    confidence=0.75,
                    reasoning_logic="Conclusion drawn from inference chain",
                    dependencies=[f"chain_{idx}_inference"],
                )
            )

            chains.append(
                ReasoningChain(
                    chain_id=f"finding_chain_{idx}",
                    question_id="",
                    steps=steps,
                    final_conclusion=finding[:200],
                    overall_confidence=0.7,
                    chain_validity="valid" if len(steps) >= 2 else "inconclusive",
                    missing_premises=[]
                    if relevant_sources
                    else ["supporting_evidence"],
                )
            )

        return chains

    def _empty_review(self) -> ContentReviewPayload:
        return ContentReviewPayload(
            resolved_findings=[],
            conflict_resolutions=[],
            remaining_gaps=[],
            confidence_adjustment=0.0,
        )


__all__ = ["ResearchContentStep"]
