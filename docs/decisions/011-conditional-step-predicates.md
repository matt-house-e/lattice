# 011: Conditional Step Execution via run_if/skip_if Predicates

**Status:** Accepted
**Date:** 2026-02-21

**Context:** Lattice users frequently need to skip steps for certain rows based on data conditions (e.g. "only run credit check for US companies", "skip deep research if classifier said 'irrelevant'"). Without this, every step runs for every row unconditionally, wasting API calls and cost. This is the first feature in Phase 6B (power user features).

**Decision:** Add `run_if` and `skip_if` optional predicate parameters to both `LLMStep` and `FunctionStep`. Predicates have signature `(row: dict, prior_results: dict) -> bool` and are evaluated per-row inside `_execute_step()` before the cache check. They are mutually exclusive (validated at construction time). Skipped rows produce field spec `default` values where available, else `None`. Skipped rows never hit cache or call `step.run()`. `RowCompleteEvent` gains a `skipped: bool` flag and `StepUsage` gains a `rows_skipped: int` counter. Both sync and async predicates are supported via `inspect.isawaitable()`. Predicate exceptions are treated as row errors by the existing `return_exceptions=True` machinery. The Step protocol is unchanged — `run_if`/`skip_if` are read via `getattr()`, so custom steps are unaffected.

**Alternatives considered:** (1) A `StepContext`-based predicate signature `(ctx: StepContext) -> bool` — rejected because `StepContext` includes `fields` and `config` which are irrelevant to skip decisions, and it prevents simple lambda usage. (2) A separate `ConditionalStep` wrapper — rejected because it adds indirection and doesn't compose well with caching/hooks. (3) Step-level skip (all rows or none) — rejected because per-row granularity is the common use case. (4) Storing skip decisions in cache — rejected because the predicate is deterministic from inputs, so caching adds overhead with no benefit.
