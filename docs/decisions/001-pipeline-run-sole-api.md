# 001: Pipeline.run() as Sole Public API

**Status:** Accepted
**Date:** 2026-01-15

**Context:** In v0.2, `Enricher` was the primary public API. Users instantiated `Enricher` directly and called `enricher.run(df)`. This mixed concerns: orchestration (how steps connect), configuration (concurrency, retries), and execution (DataFrame handling, checkpointing) were all entangled in one class. Users had to understand Enricher internals to use Lattice. Adding a Pipeline concept for step composition created a "which one do I use?" problem.

**Decision:** `Pipeline.run(df)` is the ONE public entry point. Pipeline defines topology (steps and their dependencies). Enricher is demoted to an internal runner that handles execution mechanics (DataFrame conversion, `__` field filtering, checkpointing, concurrency). Power users who need config reuse across runs can access `pipeline.runner()`, but it's not the default path. This follows "one way to do it" — users define a pipeline, run it. No choice paralysis.

**Alternatives considered:**
- **Keep Enricher public alongside Pipeline.run()** — Creates confusing dual entry points. Users would be uncertain which to use and when. Documentation burden doubles.
- **Pass all execution config through Pipeline.__init__()** — Makes Pipeline a god object holding both topology and runtime config. Inflexible: can't run the same pipeline with different concurrency settings without recreating it.
- **Builder pattern (Pipeline.build().with_config().run())** — Overly ceremonial for the common case. Most users want `Pipeline([steps]).run(df)` and nothing else.
