from __future__ import annotationsfrom typing import TYPE_CHECKINGfrom typing_extensions import overridefrom serpsage.models.pipeline import ResearchStepContextfrom serpsage.steps.base import StepBasefrom serpsage.steps.research.utils import merge_strings, normalize_stringsfrom serpsage.utils import clean_whitespaceif TYPE_CHECKING:
    from serpsage.core.runtime import Runtime

class ResearchDecideStep(StepBase[ResearchStepContext]):

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext
    ) -> ResearchStepContext:
        if ctx.current_round is None:
            return ctx

        budget = ctx.runtime.budget
        round_state = ctx.current_round
        abstract_stop = bool(ctx.work.abstract_review.stop)
        content_stop = bool(ctx.work.content_review.stop)
        strategy = str(round_state.query_strategy or "").strip().casefold()
        model_stop = bool(abstract_stop or content_stop or strategy == "stop-ready")
        confidence_ok = float(round_state.confidence) >= float(budget.stop_confidence)
        coverage_ok = float(round_state.coverage_ratio) >= float(
            budget.min_coverage_ratio
        )
        conflict_ok = int(round_state.unresolved_conflicts) <= int(
            budget.max_unresolved_conflicts
        )
        gaps_ok = int(round_state.critical_gaps) == 0
        required_entities = normalize_strings(
            ctx.plan.theme_plan.required_entities,
            limit=24,
        )
        entity_coverage_ok = (
            bool(round_state.entity_coverage_complete)
            if required_entities
            else True
        )
        multi_signal_stop = bool(
            model_stop
            and confidence_ok
            and coverage_ok
            and conflict_ok
            and gaps_ok
            and entity_coverage_ok
        )
        corpus_score_gain = float(round_state.corpus_score_gain)

        if ctx.rounds:
            prev = ctx.rounds[-1]
            progress = bool(
                round_state.new_source_ids
                or float(round_state.coverage_ratio) > float(prev.coverage_ratio)
                or int(round_state.unresolved_conflicts)
                < int(prev.unresolved_conflicts)
                or corpus_score_gain > 0.0
            )
        else:
            progress = bool(round_state.new_source_ids or corpus_score_gain > 0.0)

        if progress:
            ctx.runtime.no_progress_rounds = 0
        else:
            ctx.runtime.no_progress_rounds += 1

        raw_next_queries = merge_strings(
            list(ctx.work.next_queries),
            normalize_strings(
                ctx.work.abstract_review.next_queries,
                limit=int(budget.max_queries_per_round),
            ),
            normalize_strings(
                ctx.work.content_review.next_queries,
                limit=int(budget.max_queries_per_round),
            ),
            limit=int(budget.max_queries_per_round),
        )
        next_queries = list(raw_next_queries)
        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)
        if required_entities and not entity_coverage_ok:
            next_queries = merge_strings(
                self._build_entity_backfill_queries(
                    core_question=core_question,
                    missing_entities=round_state.missing_entities,
                    limit=int(budget.max_queries_per_round),
                ),
                next_queries,
                limit=int(budget.max_queries_per_round),
            )
        allow_auto_seed = (
            not multi_signal_stop
            and int(ctx.runtime.no_progress_rounds)
            < int(self.settings.research.no_progress_rounds_to_stop)
            and int(ctx.runtime.search_calls) < int(budget.max_search_calls)
            and int(ctx.runtime.fetch_calls) < int(budget.max_fetch_calls)
        )
        if allow_auto_seed and not next_queries and core_question:
            next_queries = [core_question]
        if (
            not multi_signal_stop
            and not next_queries
            and progress
            and int(ctx.runtime.search_calls) < int(budget.max_search_calls)
        ):
            next_queries = [ctx.request.themes]

        stop = False
        stop_reason = ""
        if multi_signal_stop:
            stop = True
            stop_reason = "multi_signal_stop"
        elif int(ctx.runtime.no_progress_rounds) >= int(
            self.settings.research.no_progress_rounds_to_stop
        ):
            stop = True
            stop_reason = "no_progress"
        elif int(ctx.runtime.search_calls) >= int(budget.max_search_calls):
            stop = True
            stop_reason = "max_search_calls"
        elif int(ctx.runtime.fetch_calls) >= int(budget.max_fetch_calls):
            stop = True
            stop_reason = "max_fetch_calls"
        elif not next_queries:
            stop = True
            stop_reason = "no_next_queries"

        round_state.stop = bool(stop)
        round_state.stop_reason = stop_reason
        ctx.plan.next_queries = list(next_queries)
        ctx.rounds.append(round_state)
        if stop:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = stop_reason

        return ctx

    def _build_entity_backfill_queries(
        self,
        *,
        core_question: str,
        missing_entities: list[str],
        limit: int,
    ) -> list[str]:
        base = clean_whitespace(core_question)
        queries: list[str] = []
        for item in normalize_strings(missing_entities, limit=16):
            entity = clean_whitespace(item)
            if not entity:
                continue
            if base:
                queries.append(f"{base} {entity}")
            else:
                queries.append(entity)
        if base:
            queries.append(base)
        return merge_strings(queries, [], limit=max(1, int(limit)))


__all__ = ["ResearchDecideStep"]
