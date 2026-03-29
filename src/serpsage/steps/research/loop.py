from __future__ import annotations

import math
from typing_extensions import override

import anyio
from anyio import CapacityLimiter, Condition, create_task_group, move_on_after

from serpsage.components.llm.base import LLMClientBase
from serpsage.dependencies import (
    RESEARCH_ROUND_RUNNER,
    RESEARCH_SUBREPORT_STEP,
    Depends,
)
from serpsage.models.app.response import ResearchResponse
from serpsage.models.steps.research import (
    GlobalBudget,
    ResearchKnowledge,
    ResearchLimits,
    ResearchResult,
    ResearchRound,
    ResearchRun,
    ResearchStepContext,
    ResearchTrackResult,
    ResearchTrackRuntime,
    RoundState,
    RoundStepContext,
    TrackAllocation,
)
from serpsage.models.steps.research.payloads import (
    ResearchThemePlanCard,
    TrackInsightCardPayload,
)
from serpsage.steps.base import RunnerBase, StepBase


class BudgetLockManager:
    """Budget/knowledge lock manager with reservation accounting.

    Lock Acquisition Order:
    1. search_lock
    2. fetch_lock
    3. knowledge_lock
    """

    __slots__ = (
        "_fetch_lock",
        "_knowledge_lock",
        "_reserved_fetch",
        "_reserved_search",
        "_search_lock",
    )

    def __init__(self) -> None:
        self._search_lock = anyio.Lock()
        self._fetch_lock = anyio.Lock()
        self._knowledge_lock = anyio.Lock()
        self._reserved_search = 0
        self._reserved_fetch = 0

    @property
    def search_lock(self) -> anyio.Lock:
        """Get the search budget lock."""
        return self._search_lock

    @property
    def fetch_lock(self) -> anyio.Lock:
        """Get the fetch budget lock."""
        return self._fetch_lock

    @property
    def knowledge_lock(self) -> anyio.Lock:
        """Get the knowledge merge lock."""
        return self._knowledge_lock

    def search_available(self, budget: GlobalBudget) -> int:
        return max(
            0,
            budget.total_search - budget.search_used - self._reserved_search,
        )

    def fetch_available(self, budget: GlobalBudget) -> int:
        return max(
            0,
            budget.total_fetch - budget.fetch_used - self._reserved_fetch,
        )

    def reserve(self, *, search: int, fetch: int) -> None:
        self._reserved_search += max(0, search)
        self._reserved_fetch += max(0, fetch)

    def release(self, *, search: int, fetch: int) -> None:
        self._reserved_search = max(0, self._reserved_search - max(0, search))
        self._reserved_fetch = max(0, self._reserved_fetch - max(0, fetch))

    def settle(
        self,
        *,
        budget: GlobalBudget,
        reserved_search: int,
        reserved_fetch: int,
        used_search: int,
        used_fetch: int,
    ) -> None:
        self.release(search=reserved_search, fetch=reserved_fetch)
        budget.search_used += max(0, min(used_search, reserved_search))
        budget.fetch_used += max(0, min(used_fetch, reserved_fetch))


class BudgetReclamationManager:
    """Event-based budget reclamation system.

    Uses anyio.Condition to signal and wait for budget reclamation:
    - Tracks signal completion via signal_completion()
    - Needy tracks request their share via request_reclaimed_budget()
    - Each requesting track gets a fair share of available budget
    - Waiting tracks are tracked to ensure fair distribution
    """

    __slots__ = (
        "_condition",
        "_reclaimable_fetch",
        "_reclaimable_search",
        "_waiting_tracks",
    )

    def __init__(self) -> None:
        self._condition = Condition()
        self._reclaimable_search = 0
        self._reclaimable_fetch = 0
        self._waiting_tracks: set[str] = set()

    async def signal_completion(
        self,
        question_id: str,
        reclaimable_search: int,
        reclaimable_fetch: int,
    ) -> None:
        """Signal that a track has completed with reclaimable budget."""
        _ = question_id
        async with self._condition:
            self._reclaimable_search += reclaimable_search
            self._reclaimable_fetch += reclaimable_fetch
            self._condition.notify_all()

    async def signal_progress(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    async def request_reclaimed_budget(
        self,
        question_id: str,
        timeout: float = 2.0,
    ) -> tuple[int, int]:
        """Request a share of reclaimed budget.

        This method ensures fair distribution among waiting tracks:
        - Each waiting track gets an equal share of available budget
        - Budget is reserved for the requester immediately
        - If no budget available, waits for signal with timeout

        Args:
            question_id: The track requesting budget.
            timeout: Max time to wait for reclamation.

        Returns:
            A tuple of (search, fetch) reclaimed amounts for this track.
        """
        async with self._condition:
            # Register as waiting
            self._waiting_tracks.add(question_id)

            # Wait if nothing available
            if self._reclaimable_search <= 0 and self._reclaimable_fetch <= 0:
                with move_on_after(timeout):
                    await self._condition.wait()

            # Calculate this track's share
            num_waiters = len(self._waiting_tracks)
            if num_waiters <= 0:
                return (0, 0)

            # Fair share: divide equally among waiters
            search_share = self._reclaimable_search // num_waiters
            fetch_share = self._reclaimable_fetch // num_waiters

            # Reserve this track's share
            self._reclaimable_search -= search_share
            self._reclaimable_fetch -= fetch_share

            # Remove from waiting list
            self._waiting_tracks.discard(question_id)

            return (search_share, fetch_share)

    @property
    def has_reclaimable(self) -> bool:
        """Check if there is reclaimable budget available."""
        return self._reclaimable_search > 0 or self._reclaimable_fetch > 0

    async def wait_for_progress(self, timeout: float = 2.0) -> None:
        async with self._condition:
            with move_on_after(timeout):
                await self._condition.wait()


def _merge_track_knowledge(
    *,
    target: ResearchKnowledge,
    source: ResearchKnowledge,
    next_source_id: int,
) -> tuple[int, dict[int, int]]:
    """Merge track's knowledge into global knowledge with ID remapping.

    This function takes sources from a track's local knowledge and merges them
    into the global knowledge, assigning new unique source IDs to avoid collisions.

    Args:
        target: The global ResearchKnowledge to merge into.
        source: The track's local ResearchKnowledge to merge from.
        next_source_id: The next available global source ID.

    Returns:
        A tuple of (count of new sources added, mapping of old_id -> new_id).
    """
    from serpsage.models.steps.research import ResearchSource

    if not source.sources:
        return 0, {}

    # Build existing URL set for deduplication
    existing_canonical_urls: set[str] = set(target.source_ids_by_url.keys())

    # Build mapping from old source_id to new source_id
    id_mapping: dict[int, int] = {}
    current_id = next_source_id
    new_sources: list[ResearchSource] = []

    for src_source in source.sources:
        canonical_url = src_source.canonical_url
        if not canonical_url:
            continue

        # Skip if this URL already exists in global knowledge
        if canonical_url in existing_canonical_urls:
            # Still need to map the ID for any references
            # Use the existing source ID from target
            existing_ids = target.source_ids_by_url.get(canonical_url, [])
            if existing_ids:
                id_mapping[src_source.source_id] = existing_ids[0]
            continue

        # Assign new ID and add to new sources
        new_source_id = current_id
        current_id += 1
        id_mapping[src_source.source_id] = new_source_id

        # Create new source with remapped ID
        new_source = src_source.model_copy(
            update={
                "source_id": new_source_id,
            },
            deep=True,
        )
        new_sources.append(new_source)

        # Update URL index
        if canonical_url not in target.source_ids_by_url:
            target.source_ids_by_url[canonical_url] = []
        target.source_ids_by_url[canonical_url].append(new_source_id)

    # Add new sources to target
    target.sources.extend(new_sources)

    # Merge covered_subthemes (preserve order, avoid duplicates)
    seen_subthemes: set[str] = {s.casefold() for s in target.covered_subthemes}
    for subtheme in source.covered_subthemes:
        if subtheme.casefold() not in seen_subthemes:
            target.covered_subthemes.append(subtheme)
            seen_subthemes.add(subtheme.casefold())

    # Note: ranked_source_ids and source_scores will be rebuilt by rebuild_corpus_ranking
    # We don't merge them here to avoid complexity with ID remapping

    return len(new_sources), id_mapping


class ResearchLoopStep(StepBase[ResearchStepContext]):
    """Parallel research track execution with intelligent budget scheduling.

    This is the ONLY class that manages RoundStepContext instances.
    Round runner steps only see RoundStepContext, never ResearchStepContext.
    """

    _EXTENSION_MULTIPLIERS: dict[str, float] = {
        "low": 0.2,
        "medium": 0.4,
        "high": 0.6,
    }

    llm: LLMClientBase = Depends()
    round_runner: RunnerBase[RoundStepContext] = Depends(RESEARCH_ROUND_RUNNER)
    render_step: StepBase[ResearchStepContext] = Depends(RESEARCH_SUBREPORT_STEP)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        question_cards = self._resolve_question_cards(ctx)
        ctx.result.tracks = []
        if not question_cards:
            ctx.run.stop = True
            ctx.run.stop_reason = "no_question_cards"
            return ctx

        # Initialize global budget
        ctx.run.budget = GlobalBudget(
            total_search=ctx.run.limits.max_search_calls,
            total_fetch=ctx.run.limits.max_fetch_calls,
        )

        # Initialize lock manager and reclamation manager
        lock_manager = BudgetLockManager()
        reclamation_manager = BudgetReclamationManager()

        # Build track contexts (RoundStepContext, not ResearchStepContext)
        track_contexts = self._build_track_contexts(ctx, question_cards)

        # Initialize budget allocation with minimum guarantees
        self._init_budget_allocation(ctx, question_cards, track_contexts)

        # Execute tracks in parallel
        results = await self._run_parallel(
            ctx=ctx,
            track_contexts=track_contexts,
            lock_manager=lock_manager,
            reclamation_manager=reclamation_manager,
        )

        # Aggregate results
        ctx.result.tracks = [
            results[card.question_id]
            for card in question_cards
            if card.question_id in results
        ]
        ctx.run.stop = True
        ctx.run.stop_reason = self._determine_stop_reason(results, question_cards)
        return ctx

    def _init_budget_allocation(
        self,
        ctx: ResearchStepContext,
        cards: list[ResearchThemePlanCard],
        track_contexts: dict[str, RoundStepContext],
    ) -> None:
        """Initialize budget allocation based on priority with minimum guarantees."""
        allocations = self._allocate_by_priority(
            cards=cards,
            total_search=ctx.run.budget.total_search,
            total_fetch=ctx.run.budget.total_fetch,
            limits=ctx.run.limits,
        )

        for card in cards:
            alloc = allocations[card.question_id]
            runtime = ResearchTrackRuntime(
                question_id=card.question_id,
                priority=card.priority,
                state=(
                    "active"
                    if alloc["search"] > 0 or alloc["fetch"] > 0
                    else "waiting_for_budget"
                ),
            )
            runtime.allocation.search_quota = alloc["search"]
            runtime.allocation.fetch_quota = alloc["fetch"]
            runtime.allocation.minimum_search = alloc["minimum_search"]
            runtime.allocation.minimum_fetch = alloc["minimum_fetch"]
            ctx.run.track_runtimes[card.question_id] = runtime

            # Set track allocation (no longer mutating limits!)
            track_ctx = track_contexts[card.question_id]
            track_ctx.run.allocation = TrackAllocation(
                search_quota=alloc["search"],
                fetch_quota=alloc["fetch"],
                minimum_search=alloc["minimum_search"],
                minimum_fetch=alloc["minimum_fetch"],
            )
            track_ctx.run.limits.max_queries_per_round = max(
                1,
                min(
                    ctx.run.limits.max_queries_per_round,
                    ctx.run.limits.round_search_budget,
                    alloc["search"],
                ),
            )

    def _allocate_by_priority(
        self,
        cards: list[ResearchThemePlanCard],
        total_search: int,
        total_fetch: int,
        limits: ResearchLimits,
    ) -> dict[str, dict[str, int]]:
        """Allocate exact quotas without overcommitting unavailable global budget."""
        if not cards:
            return {}

        search_allocations, search_floors = self._allocate_resource_by_priority(
            cards=cards,
            total=total_search,
            protected_cap=limits.round_search_budget,
        )
        fetch_allocations, fetch_floors = self._allocate_resource_by_priority(
            cards=cards,
            total=total_fetch,
            protected_cap=limits.round_fetch_budget,
        )

        return {
            card.question_id: {
                "search": search_allocations[card.question_id],
                "fetch": fetch_allocations[card.question_id],
                "minimum_search": min(
                    search_allocations[card.question_id],
                    search_floors[card.question_id],
                ),
                "minimum_fetch": min(
                    fetch_allocations[card.question_id],
                    fetch_floors[card.question_id],
                ),
            }
            for card in cards
        }

    def _allocate_resource_by_priority(
        self,
        *,
        cards: list[ResearchThemePlanCard],
        total: int,
        protected_cap: int,
    ) -> tuple[dict[str, int], dict[str, int]]:
        allocations = {card.question_id: 0 for card in cards}
        protected_floors = {card.question_id: 0 for card in cards}
        if total <= 0 or not cards:
            return allocations, protected_floors

        card_count = len(cards)
        base_floor = min(max(0, protected_cap), total // card_count)
        if base_floor > 0:
            for card in cards:
                allocations[card.question_id] = base_floor
                protected_floors[card.question_id] = base_floor

        remaining = total - (base_floor * card_count)
        if remaining <= 0:
            return allocations, protected_floors

        total_priority = sum(card.priority for card in cards)
        if total_priority <= 0:
            total_priority = card_count

        fractional_parts: dict[str, float] = {}
        original_order = {card.question_id: index for index, card in enumerate(cards)}
        for card in cards:
            exact_extra = (remaining * card.priority) / total_priority
            whole_extra = int(exact_extra)
            allocations[card.question_id] += whole_extra
            fractional_parts[card.question_id] = exact_extra - whole_extra

        remainder = total - sum(allocations.values())
        if remainder <= 0:
            return allocations, protected_floors

        ordered_cards = sorted(
            cards,
            key=lambda card: (
                fractional_parts[card.question_id],
                card.priority,
                -original_order[card.question_id],
            ),
            reverse=True,
        )
        for card in ordered_cards[:remainder]:
            allocations[card.question_id] += 1

        return allocations, protected_floors

    async def _reserve_round_budget(
        self,
        ctx: ResearchStepContext,
        question_id: str,
        lock_manager: BudgetLockManager,
    ) -> tuple[int, int] | None:
        """Reserve round budget for a track.

        Returns the per-round search/fetch grant after accounting for:
        - track-local remaining quota
        - globally available quota minus in-flight reservations
        - per-round caps
        """
        runtime = ctx.run.track_runtimes.get(question_id)
        if not runtime or runtime.state in {"completed", "stopped"}:
            return None

        async with lock_manager.search_lock, lock_manager.fetch_lock:
            global_search_available = lock_manager.search_available(ctx.run.budget)
            global_fetch_available = lock_manager.fetch_available(ctx.run.budget)
            track_search_remaining = runtime.allocation.search_remaining
            track_fetch_remaining = runtime.allocation.fetch_remaining

            if global_search_available <= 0 and global_fetch_available <= 0:
                return None
            if track_search_remaining <= 0 and track_fetch_remaining <= 0:
                return None

            round_search = min(
                track_search_remaining,
                global_search_available,
                ctx.run.limits.round_search_budget,
            )
            round_fetch = min(
                track_fetch_remaining,
                global_fetch_available,
                ctx.run.limits.round_fetch_budget,
            )
            if round_search <= 0 and round_fetch <= 0:
                return None

            reserved_search = max(0, round_search)
            reserved_fetch = max(0, round_fetch)
            lock_manager.reserve(search=reserved_search, fetch=reserved_fetch)
            return (reserved_search, reserved_fetch)

    async def _try_reclaim_budget(
        self,
        *,
        ctx: ResearchStepContext,
        question_id: str,
        reclamation_manager: BudgetReclamationManager,
        lock_manager: BudgetLockManager,
    ) -> bool:
        runtime = ctx.run.track_runtimes.get(question_id)
        if runtime is None:
            return False
        reclaimed = await reclamation_manager.request_reclaimed_budget(
            question_id=question_id,
            timeout=2.0,
        )
        if reclaimed[0] <= 0 and reclaimed[1] <= 0:
            return False
        async with lock_manager.search_lock, lock_manager.fetch_lock:
            runtime.allocation.search_quota += reclaimed[0]
            runtime.allocation.fetch_quota += reclaimed[1]
            runtime.budget_tier = ctx.run.budget.tier
        return True

    def _has_peer_budget_path(
        self,
        *,
        ctx: ResearchStepContext,
        question_id: str,
    ) -> bool:
        for peer_id, runtime in ctx.run.track_runtimes.items():
            if peer_id == question_id or runtime.state in {"completed", "stopped"}:
                continue
            if (
                runtime.allocation.search_remaining > 0
                or runtime.allocation.fetch_remaining > 0
            ):
                return True
        return False

    def _settle_round_budget(
        self,
        ctx: ResearchStepContext,
        question_id: str,
        reserved_search: int,
        reserved_fetch: int,
        used_search: int,
        used_fetch: int,
        lock_manager: BudgetLockManager,
    ) -> None:
        """Commit actual usage and release any unused in-flight reservation."""
        lock_manager.settle(
            budget=ctx.run.budget,
            reserved_search=reserved_search,
            reserved_fetch=reserved_fetch,
            used_search=used_search,
            used_fetch=used_fetch,
        )

        runtime = ctx.run.track_runtimes.get(question_id)
        if runtime:
            runtime.allocation.search_used += max(0, min(used_search, reserved_search))
            runtime.allocation.fetch_used += max(0, min(used_fetch, reserved_fetch))

    def _apply_budget_escalation(self, ctx: ResearchStepContext) -> bool:
        """Apply budget escalation when tracks are stuck.

        NOTE: Caller MUST hold both search_lock and fetch_lock before calling
        this method to ensure thread-safe quota modifications.
        Lock order: search_lock first, then fetch_lock.
        """
        if ctx.run.budget.tier != "base":
            return False

        multiplier = self._EXTENSION_MULTIPLIERS.get(ctx.task.complexity, 0.4)
        original_search = ctx.run.limits.max_search_calls
        original_fetch = ctx.run.limits.max_fetch_calls

        extension_search = max(1, math.ceil(original_search * multiplier))
        extension_fetch = max(1, math.ceil(original_fetch * multiplier))

        ctx.run.budget.total_search += extension_search
        ctx.run.budget.total_fetch += extension_fetch
        ctx.run.budget.tier = "extension"

        # Distribute to all unfinished tracks so waiting tracks can resume too.
        unfinished = [
            runtime
            for runtime in ctx.run.track_runtimes.values()
            if runtime.state not in {"completed", "stopped"}
        ]
        if unfinished:
            per_track_search = extension_search // len(unfinished)
            per_track_fetch = extension_fetch // len(unfinished)
            search_remainder = extension_search % len(unfinished)
            fetch_remainder = extension_fetch % len(unfinished)
            for index, runtime in enumerate(unfinished):
                runtime.allocation.search_quota += per_track_search + (
                    1 if index < search_remainder else 0
                )
                runtime.allocation.fetch_quota += per_track_fetch + (
                    1 if index < fetch_remainder else 0
                )
                runtime.budget_tier = "extension"

        return True

    # ========== Parallel Execution ==========

    async def _run_parallel(
        self,
        ctx: ResearchStepContext,
        track_contexts: dict[str, RoundStepContext],
        lock_manager: BudgetLockManager,
        reclamation_manager: BudgetReclamationManager,
    ) -> dict[str, ResearchTrackResult]:
        """Execute all tracks in parallel."""
        results: dict[str, ResearchTrackResult] = {}
        limiter = CapacityLimiter(ctx.run.limits.max_concurrent_tracks)

        async def run_track(question_id: str, track_ctx: RoundStepContext) -> None:
            async with limiter:
                try:
                    result = await self._execute_track(
                        ctx=ctx,
                        question_id=question_id,
                        track_ctx=track_ctx,
                        lock_manager=lock_manager,
                        reclamation_manager=reclamation_manager,
                    )
                    results[question_id] = result
                except Exception:
                    await reclamation_manager.signal_completion(
                        question_id=question_id,
                        reclaimable_search=0,
                        reclaimable_fetch=0,
                    )
                    raise

        async with create_task_group() as tg:
            for qid, tctx in track_contexts.items():
                tg.start_soon(run_track, qid, tctx)

        return results

    async def _execute_track(
        self,
        ctx: ResearchStepContext,
        question_id: str,
        track_ctx: RoundStepContext,
        lock_manager: BudgetLockManager,
        reclamation_manager: BudgetReclamationManager,
    ) -> ResearchTrackResult:
        """Execute a single track with budget management."""
        runtime = ctx.run.track_runtimes.get(question_id)
        if not runtime:
            runtime = ResearchTrackRuntime(question_id=question_id)
            ctx.run.track_runtimes[question_id] = runtime

        while True:
            budget = await self._reserve_round_budget(
                ctx=ctx,
                question_id=question_id,
                lock_manager=lock_manager,
            )

            if budget is None:
                if runtime.allocation.is_exhausted and await self._try_reclaim_budget(
                    ctx=ctx,
                    question_id=question_id,
                    reclamation_manager=reclamation_manager,
                    lock_manager=lock_manager,
                ):
                    continue

                # Try escalation only when the global pool itself is exhausted.
                escalated = False
                async with lock_manager.search_lock, lock_manager.fetch_lock:
                    global_exhausted = (
                        lock_manager.search_available(ctx.run.budget) <= 0
                        and lock_manager.fetch_available(ctx.run.budget) <= 0
                    )
                    if global_exhausted:
                        escalated = self._apply_budget_escalation(ctx)
                if escalated:
                    await reclamation_manager.signal_progress()
                    continue

                if runtime.allocation.is_exhausted and self._has_peer_budget_path(
                    ctx=ctx,
                    question_id=question_id,
                ):
                    if runtime.state != "waiting_for_budget":
                        runtime.waiting_rounds += 1
                    runtime.state = "waiting_for_budget"
                    runtime.stop_reason = "waiting_for_budget"
                    if track_ctx.run.current is not None:
                        track_ctx.run.current.waiting_for_budget = True
                        track_ctx.run.current.waiting_reason = (
                            track_ctx.run.current.waiting_reason
                            or "budget_resume_required"
                        )
                    await reclamation_manager.wait_for_progress(timeout=2.0)
                    continue

                runtime.state = "stopped"
                runtime.stop_reason = "budget_exhausted"
                track_ctx.run.stop = True
                track_ctx.run.stop_reason = "budget_exhausted"
                track_ctx.run.archive_current_round(
                    stop=True,
                    stop_reason="budget_exhausted",
                )
                break

            track_ctx.run.allocation = runtime.allocation.model_copy(
                update={
                    "search_quota": min(
                        runtime.allocation.search_quota,
                        runtime.allocation.search_used + budget[0],
                    ),
                    "fetch_quota": min(
                        runtime.allocation.fetch_quota,
                        runtime.allocation.fetch_used + budget[1],
                    ),
                },
                deep=True,
            )
            track_ctx.run.limits.max_queries_per_round = max(
                1,
                min(ctx.run.limits.max_queries_per_round, max(1, budget[0])),
            )
            track_ctx.run.limits.round_fetch_budget = max(1, budget[1])

            before_search = track_ctx.run.allocation.search_used
            before_fetch = track_ctx.run.allocation.fetch_used
            before_explore_links = track_ctx.run.explore_resolved_relative_links

            track_ctx.run.stop = False
            track_ctx.run.stop_reason = ""
            if track_ctx.run.current is not None:
                track_ctx.run.current.stop = False
                track_ctx.run.current.stop_reason = ""

            try:
                track_ctx = await self.round_runner.run(track_ctx)
            except Exception as exc:
                await self.tracker.error(
                    name="research.loop.track_failed",
                    request_id=ctx.request_id,
                    step="research.loop",
                    error_code="research_loop_track_failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    data={"question_id": question_id},
                )
                delta_search = max(
                    0,
                    track_ctx.run.allocation.search_used - before_search,
                )
                delta_fetch = max(
                    0,
                    track_ctx.run.allocation.fetch_used - before_fetch,
                )
                async with lock_manager.search_lock, lock_manager.fetch_lock:
                    self._settle_round_budget(
                        ctx=ctx,
                        question_id=question_id,
                        reserved_search=budget[0],
                        reserved_fetch=budget[1],
                        used_search=delta_search,
                        used_fetch=delta_fetch,
                        lock_manager=lock_manager,
                    )
                raise

            delta_search = max(0, track_ctx.run.allocation.search_used - before_search)
            delta_fetch = max(0, track_ctx.run.allocation.fetch_used - before_fetch)
            delta_explore_links = max(
                0,
                track_ctx.run.explore_resolved_relative_links - before_explore_links,
            )
            async with lock_manager.search_lock, lock_manager.fetch_lock:
                self._settle_round_budget(
                    ctx=ctx,
                    question_id=question_id,
                    reserved_search=budget[0],
                    reserved_fetch=budget[1],
                    used_search=delta_search,
                    used_fetch=delta_fetch,
                    lock_manager=lock_manager,
                )
            track_ctx.run.allocation = runtime.allocation.model_copy(deep=True)
            ctx.run.explore_resolved_relative_links += delta_explore_links

            runtime.completed_rounds = len(track_ctx.run.history)

            if track_ctx.run.stop:
                runtime.state = "completed"
                runtime.stop_reason = track_ctx.run.stop_reason or "completed"
                await reclamation_manager.signal_completion(
                    question_id=question_id,
                    reclaimable_search=runtime.allocation.reclaimable_search,
                    reclaimable_fetch=runtime.allocation.reclaimable_fetch,
                )
                break

            if self._track_is_waiting(track_ctx):
                if runtime.state != "waiting_for_budget":
                    runtime.waiting_rounds += 1
                runtime.state = "waiting_for_budget"
                runtime.stop_reason = track_ctx.run.stop_reason or "waiting_for_budget"
            else:
                runtime.state = "active"
                runtime.waiting_rounds = 0
                runtime.stop_reason = ""

        return await self._finalize_track(
            ctx=ctx,
            question_id=question_id,
            track_ctx=track_ctx,
            runtime=runtime,
            lock_manager=lock_manager,
        )

    def _track_is_waiting(self, track_ctx: RoundStepContext) -> bool:
        """Check if track is waiting for budget."""
        current = track_ctx.run.current
        if current is None:
            return False
        return current.needs_resume

    # ========== Helper Methods ==========

    def _resolve_question_cards(
        self, ctx: ResearchStepContext
    ) -> list[ResearchThemePlanCard]:
        return [
            item.model_copy(deep=True)
            for item in ctx.task.cards[
                : max(1, ctx.run.limits.max_question_cards_effective)
            ]
        ]

    def _build_track_contexts(
        self,
        ctx: ResearchStepContext,
        cards: list[ResearchThemePlanCard],
    ) -> dict[str, RoundStepContext]:
        """Build RoundStepContext instances for each track.

        This is the ONLY place where RoundStepContext is created.
        """
        track_contexts: dict[str, RoundStepContext] = {}
        for card in cards:
            track_contexts[card.question_id] = self._build_track_context(
                root=ctx, card=card
            )
        return track_contexts

    def _build_track_context(
        self,
        root: ResearchStepContext,
        card: ResearchThemePlanCard,
    ) -> RoundStepContext:
        """Build a single RoundStepContext for a track.

        Data is copied IN from ResearchStepContext.
        Data will be copied OUT in _finalize_track.
        """
        request = root.request.model_copy(
            update={"themes": card.question, "json_schema": None}
        )
        track = RoundStepContext(
            request=request,
            request_id=f"{root.request_id}:track:{card.question_id}",
            response=ResearchResponse(
                request_id=f"{root.request_id}:track:{card.question_id}",
                content="",
                structured=None,
            ),
            question_id=card.question_id,
            task=root.task.model_copy(deep=True),
            run=RoundState(
                limits=root.run.limits.model_copy(deep=True),
                allocation=TrackAllocation(),
                stop=False,
                stop_reason="",
                round_index=0,
                next_queries=[q.model_copy(deep=True) for q in card.seed_queries],
                link_candidates={},
                link_candidates_round=0,
                notes=[f"Track initialized for question `{card.question_id}`."],
                current=None,
                history=[],
            ),
            knowledge=root.knowledge.model_copy(deep=True),
        )
        track.task.question = card.question
        track.task.cards = [card.model_copy(deep=True)]
        return track

    def _determine_stop_reason(
        self,
        results: dict[str, ResearchTrackResult],
        cards: list[ResearchThemePlanCard],
    ) -> str:
        budget_stop_reasons = {
            "budget_exhausted",
            "max_search_calls",
            "max_fetch_calls",
        }
        all_completed = all(
            results.get(
                card.question_id,
                ResearchTrackResult(
                    question_id=card.question_id, question=card.question
                ),
            ).stop_reason
            not in budget_stop_reasons
            for card in cards
        )
        return "all_tracks_completed" if all_completed else "budget_exhausted"

    async def _finalize_track(
        self,
        ctx: ResearchStepContext,
        question_id: str,
        track_ctx: RoundStepContext,
        runtime: ResearchTrackRuntime,
        lock_manager: BudgetLockManager,
    ) -> ResearchTrackResult:
        """Finalize a track and extract results.

        Data is copied OUT from RoundStepContext to ResearchStepContext.
        Subreport rendering uses an isolated knowledge copy to avoid cross-track
        mutation. Track knowledge is merged into the global corpus afterward.
        """
        render_ctx = self._create_render_context(
            ctx,
            track_ctx,
            knowledge=track_ctx.knowledge.model_copy(deep=True),
        )
        rendered = await self.render_step.run(render_ctx)

        async with lock_manager.knowledge_lock:
            next_source_id = len(ctx.knowledge.sources) + 1
            new_source_count, id_mapping = _merge_track_knowledge(
                target=ctx.knowledge,
                source=track_ctx.knowledge,
                next_source_id=next_source_id,
            )

            # Log merge statistics
            if new_source_count > 0:
                await self.tracker.debug(
                    name="research.loop.knowledge_merged",
                    request_id=ctx.request_id,
                    step="research.loop",
                    data={
                        "success": True,
                        "question_id": question_id,
                        "new_sources_added": new_source_count,
                        "id_mappings": len(id_mapping),
                    },
                )

        latest_round = self._latest_round(rendered)
        insight_card = (
            rendered.result.structured
            if isinstance(rendered.result.structured, TrackInsightCardPayload)
            else None
        )
        summary_rounds = self._rounds_for_summary(track_ctx.run)
        key_findings = self._extract_key_findings_from_rounds(summary_rounds)

        await self.tracker.info(
            name="research.loop.subreport_ready",
            request_id=rendered.request_id,
            step="research.loop",
            data={
                "success": True,
                "question_id": question_id,
                "rounds": len(summary_rounds),
                "stop_reason": track_ctx.run.stop_reason or "n/a",
                "has_insight_card": insight_card is not None,
                "new_sources_merged": new_source_count,
            },
        )

        return ResearchTrackResult(
            question_id=question_id,
            question=track_ctx.task.question,
            stop_reason=runtime.stop_reason or track_ctx.run.stop_reason,
            rounds=len(summary_rounds),
            search_used=runtime.allocation.search_used,
            fetch_used=runtime.allocation.fetch_used,
            confidence=self._get_round_confidence(latest_round),
            coverage_ratio=latest_round.coverage_ratio if latest_round else 0.0,
            unresolved_conflicts=self._get_round_unresolved_conflicts(latest_round),
            subreport_markdown=rendered.result.content,
            track_insight_card=insight_card,
            key_findings=key_findings,
            budget_tier=runtime.budget_tier,
            waiting_rounds=runtime.waiting_rounds,
        )

    def _create_render_context(
        self,
        root: ResearchStepContext,
        track_ctx: RoundStepContext,
        *,
        knowledge: ResearchKnowledge,
    ) -> ResearchStepContext:
        """Create a ResearchStepContext for render_step from RoundStepContext.

        Uses track_state to provide access to track execution state
        without duplicating fields in ResearchRun.
        """
        track_runtimes_copy = {
            qid: runtime.model_copy(deep=True)
            for qid, runtime in root.run.track_runtimes.items()
        }

        return ResearchStepContext(
            request=track_ctx.request,
            request_id=track_ctx.request_id,
            response=track_ctx.response,
            task=track_ctx.task,
            run=ResearchRun(
                mode=root.run.mode,
                limits=track_ctx.run.limits.model_copy(deep=True),
                budget=GlobalBudget(
                    total_search=track_ctx.run.allocation.search_quota,
                    total_fetch=track_ctx.run.allocation.fetch_quota,
                    search_used=track_ctx.run.allocation.search_used,
                    fetch_used=track_ctx.run.allocation.fetch_used,
                    tier=root.run.budget.tier,
                ),
                track_runtimes=track_runtimes_copy,  # Include all track runtimes
                stop=track_ctx.run.stop,
                stop_reason=track_ctx.run.stop_reason,
                notes=list(track_ctx.run.notes),
                explore_resolved_relative_links=track_ctx.run.explore_resolved_relative_links,
            ),
            knowledge=knowledge,
            result=ResearchResult(content="", structured=None, tracks=[]),
            track_state=track_ctx.run.model_copy(deep=True),
        )

    def _extract_key_findings_from_rounds(
        self, history: list[ResearchRound]
    ) -> list[str]:
        findings: list[str] = []
        seen: set[str] = set()

        for round_state in history[-8:]:
            for item in round_state.overview_summary.split("|"):
                token = item.strip()
                if token and token.casefold() not in seen:
                    seen.add(token.casefold())
                    findings.append(token)
            for item in round_state.content_summary.split("|"):
                token = item.strip()
                if token and token.casefold() not in seen:
                    seen.add(token.casefold())
                    findings.append(token)

        return findings[:8]

    def _rounds_for_summary(self, track_state: RoundState) -> list[ResearchRound]:
        rounds = list(track_state.history)
        if track_state.current is not None:
            rounds.append(track_state.current)
        return rounds

    def _latest_round(self, ctx: ResearchStepContext) -> ResearchRound | None:
        if ctx.track_state is None:
            return None
        return ctx.track_state.latest_round

    def _get_round_confidence(self, round_state: ResearchRound | None) -> float:
        return round_state.confidence if round_state is not None else 0.0

    def _get_round_unresolved_conflicts(self, round_state: ResearchRound | None) -> int:
        return round_state.unresolved_conflicts if round_state is not None else 0


__all__ = [
    "BudgetLockManager",
    "BudgetReclamationManager",
    "ResearchLoopStep",
]
