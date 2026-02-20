# 003: Column-Oriented Execution, Not Row-Oriented

**Status:** Accepted
**Date:** 2026-01-10

**Context:** In v0.2, Lattice processed each row through the full pipeline (all steps) before moving to the next row — row-oriented execution. This made step dependencies hard to reason about (Step 2 needed Step 1's output for the *same* row, but Step 1 hadn't run for other rows yet). Parallelizing independent steps was impossible since execution was bound to one row at a time. Caching was awkward because step context wasn't fully materialized.

**Decision:** Pipeline executes column-oriented: all rows through Step 1, then all rows through Step 2, then Step 3. Within a step, rows run concurrently (bounded by semaphore). Independent steps at the same DAG level run in parallel via `asyncio.gather`. This means Step 2 sees Step 1's complete output across ALL rows before it starts. The execution model is: for each level in the DAG, for each step in level (parallel), for each row (concurrent with semaphore).

**Alternatives considered:**
- **Row-oriented (v0.2 model)** — Can't parallelize independent steps. Harder to share cross-row context. Cache granularity is coarser (entire row pipeline vs. per-step-per-row).
- **Chunk-oriented (N rows at a time through full pipeline)** — Deferred to Phase 6B. Useful for very large datasets where materializing all rows for one step exceeds memory, but adds complexity. Caching needs to work well first (Phase 4) before chunking adds value.
- **Streaming/row-at-a-time with futures** — Complex dependency resolution, hard to reason about partial state. Column-oriented is simpler and matches the batch enrichment use case.
