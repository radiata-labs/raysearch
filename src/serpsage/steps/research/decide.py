from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import merge_strings, normalize_strings
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class ResearchDecideStep(StepBase[ResearchStepContext]):
    span_name = "step.research_decide"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        if ctx.current_round is None:
            return ctx

        budget = ctx.runtime.budget
        round_state = ctx.current_round
        abstract_stop = bool(ctx.work.abstract_review.get("stop", False))
        content_stop = bool(ctx.work.content_review.get("stop", False))
        strategy = str(round_state.query_strategy or "").strip().casefold()
        model_stop = bool(abstract_stop or content_stop or strategy == "stop-ready")
        confidence_ok = float(round_state.confidence) >= float(budget.stop_confidence)
        coverage_ok = float(round_state.coverage_ratio) >= float(budget.min_coverage_ratio)
        conflict_ok = int(round_state.unresolved_conflicts) <= int(
            budget.max_unresolved_conflicts
        )
        gaps_ok = int(round_state.critical_gaps) == 0
        multi_signal_stop = bool(
            model_stop and confidence_ok and coverage_ok and conflict_ok and gaps_ok
        )

        if ctx.rounds:
            prev = ctx.rounds[-1]
            progress = bool(
                round_state.new_source_ids
                or float(round_state.coverage_ratio) > float(prev.coverage_ratio)
                or int(round_state.unresolved_conflicts)
                < int(prev.unresolved_conflicts)
            )
        else:
            progress = bool(round_state.new_source_ids)
        round_state.progress = bool(progress)

        if progress:
            ctx.runtime.no_progress_rounds = 0
        else:
            ctx.runtime.no_progress_rounds += 1

        raw_next_queries = merge_strings(
            list(ctx.work.next_queries),
            normalize_strings(
                ctx.work.abstract_review.get("next_queries"),
                limit=int(budget.max_queries_per_round),
            ),
            normalize_strings(
                ctx.work.content_review.get("next_queries"),
                limit=int(budget.max_queries_per_round),
            ),
            limit=int(budget.max_queries_per_round),
        )
        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)
        next_queries, dropped_off_topic = self._apply_focus_guard(
            raw_next_queries,
            core_question=core_question,
            limit=int(budget.max_queries_per_round),
        )
        allow_auto_rewrite = (
            not multi_signal_stop
            and int(ctx.runtime.no_progress_rounds)
            < int(self.settings.research.no_progress_rounds_to_stop)
            and int(ctx.runtime.search_calls) < int(budget.max_search_calls)
            and int(ctx.runtime.fetch_calls) < int(budget.max_fetch_calls)
        )
        if allow_auto_rewrite and not next_queries:
            next_queries = [self._rewrite_to_core_question(core_question, raw_next_queries)]
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

        round_state.next_queries = list(next_queries)
        round_state.stop = bool(stop)
        round_state.stop_reason = stop_reason
        ctx.plan.next_queries = list(next_queries)
        ctx.rounds.append(round_state)
        if stop:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = stop_reason

        span.set_attr("round_index", int(round_state.round_index))
        span.set_attr("model_stop", bool(model_stop))
        span.set_attr("confidence_ok", bool(confidence_ok))
        span.set_attr("coverage_ok", bool(coverage_ok))
        span.set_attr("conflict_ok", bool(conflict_ok))
        span.set_attr("gaps_ok", bool(gaps_ok))
        span.set_attr("progress", bool(progress))
        span.set_attr("no_progress_rounds", int(ctx.runtime.no_progress_rounds))
        span.set_attr("dropped_off_topic_queries", int(dropped_off_topic))
        span.set_attr("stop", bool(stop))
        span.set_attr("stop_reason", stop_reason)
        print(
            "[research.decide]",
            json.dumps(
                {
                    "round_index": int(round_state.round_index),
                    "model_stop": bool(model_stop),
                    "confidence_ok": bool(confidence_ok),
                    "coverage_ok": bool(coverage_ok),
                    "conflict_ok": bool(conflict_ok),
                    "gaps_ok": bool(gaps_ok),
                    "multi_signal_stop": bool(multi_signal_stop),
                    "progress": bool(progress),
                    "no_progress_rounds": int(ctx.runtime.no_progress_rounds),
                    "dropped_off_topic_queries": int(dropped_off_topic),
                    "next_queries": list(next_queries),
                    "stop": bool(stop),
                    "stop_reason": str(stop_reason),
                },
                ensure_ascii=False,
            ),
        )
        return ctx

    def _apply_focus_guard(
        self,
        values: list[str],
        *,
        core_question: str,
        limit: int,
    ) -> tuple[list[str], int]:
        kept: list[str] = []
        dropped = 0
        for item in values:
            query = clean_whitespace(item)
            if not query:
                continue
            if not self._is_query_focused(query, core_question):
                dropped += 1
                continue
            kept.append(query)
            if len(kept) >= max(1, int(limit)):
                break
        return kept, dropped

    def _rewrite_to_core_question(
        self,
        core_question: str,
        candidates: list[str],
    ) -> str:
        core = clean_whitespace(core_question)
        if not core:
            return ""
        core_terms = self._extract_terms(core)
        best_aux = ""
        for query in candidates:
            for term in self._extract_terms(query):
                if term in core_terms or len(term) <= 1:
                    continue
                best_aux = term
                break
            if best_aux:
                break
        if not best_aux:
            return core
        return clean_whitespace(f"{core} {best_aux}")

    def _is_query_focused(self, query: str, core_question: str) -> bool:
        query_text = clean_whitespace(query).casefold()
        core_text = clean_whitespace(core_question).casefold()
        if not query_text or not core_text:
            return False
        if core_text in query_text or query_text in core_text:
            return True
        query_terms = self._extract_terms(query_text)
        core_terms = self._extract_terms(core_text)
        if not query_terms or not core_terms:
            return False
        overlap = len(query_terms & core_terms)
        core_ratio = overlap / max(1, len(core_terms))
        query_ratio = overlap / max(1, len(query_terms))
        cfg = self.settings.research.parallel
        return (
            overlap >= int(cfg.focus_min_term_overlap)
            and (
                core_ratio >= float(cfg.focus_min_overlap_ratio)
                or query_ratio >= float(cfg.focus_min_overlap_ratio)
            )
        )

    def _extract_terms(self, text: str) -> set[str]:
        cleaned = clean_whitespace(text).casefold()
        terms = {item for item in re.findall(r"[a-z0-9]+", cleaned) if item}
        for token in re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff]+", cleaned):
            if not token:
                continue
            terms.add(token)
            terms.update(ch for ch in token if ch.strip())
        return terms


__all__ = ["ResearchDecideStep"]
