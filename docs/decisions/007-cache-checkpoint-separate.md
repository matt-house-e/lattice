# 007: Cache and Checkpoint as Separate Concerns

**Status:** Accepted
**Date:** 2026-02-01

**Context:** Phase 4 introduced both caching (row-level deduplication across runs) and checkpoint enhancement (step-level crash recovery within a run). These could be merged into a single "recovery system" or kept independent. The question was whether combining them would simplify or complicate the mental model and implementation.

**Decision:** Cache and checkpoint are independent, separately configurable concerns. **Checkpoint** is step-level crash recovery: "pipeline died at step 3 of 5, resume from step 2's output." It's a fast single-file mechanism for resuming interrupted runs. **Cache** is row-level input-hash deduplication: "I changed step 3's prompt, skip steps 1 and 2 for unchanged rows." It works across runs, is content-addressed, and auto-invalidates. Config exposes both `enable_caching` and `enable_checkpointing` independently. `checkpoint_interval` saves partial step progress every N rows for large datasets.

**Alternatives considered:**
- **Merge into one "recovery system"** — Conflates different concerns. Dev workflows need cache (iterate prompts quickly). Production workflows need checkpoint (don't lose hours of work on crash). Merging forces both or neither.
- **Cache only, no checkpoint** — Large datasets (10K+ rows) lose entire step progress on crash. A step taking 30 minutes would restart from scratch.
- **Checkpoint only, no cache** — Iterative development is painful. Changing one step's prompt re-runs all LLM calls for all steps, even unchanged ones.
