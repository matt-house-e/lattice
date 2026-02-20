# 008: Lifecycle Hooks on run(), Not on Config

**Status:** Accepted
**Date:** 2026-02-10

**Context:** Issue #30 requested lifecycle hooks for observability. The question was where hooks should live: on `EnrichmentConfig` (serializable, reusable), on `Pipeline.__init__()` (fixed per pipeline), or passed to `run()` (per-invocation). Hooks are callables (functions, lambdas, async coroutines) — fundamentally different from config values like concurrency limits or model names.

**Decision:** `EnrichmentHooks` dataclass with 5 typed callbacks: `on_pipeline_start`, `on_pipeline_end`, `on_step_start`, `on_step_end`, `on_row_complete`. Passed to `pipeline.run(hooks=...)` or `pipeline.run_async(hooks=...)`. Each event is a typed dataclass (e.g., `RowCompleteEvent` with `row_index`, `values`, `error`, `from_cache`). Both sync and async hooks are supported. Hook errors are caught and logged — observability failures never crash the pipeline. This also subsumes streaming (issue #11): `on_row_complete` lets applications process results progressively.

**Alternatives considered:**
- **Hooks on config** — Config is serializable (JSON/YAML). Hooks are callables, not serializable. Mixing them conflates data and behavior. Different environments (dev logging vs. prod metrics) would need different configs just for hooks.
- **Hooks on Pipeline.__init__()** — Can't change hooks between runs without recreating the pipeline. A pipeline's topology (steps, dependencies) is fixed; its observability should be flexible.
- **Global hook registry** — Hard to test, pollutes global state, can't have different hooks for different pipelines in the same process.
- **Single on_complete callback** — Missing intermediate visibility. Step-level and row-level hooks are essential for progress tracking and debugging.
