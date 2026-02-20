# 002: Fields Live on Steps, Not in a Registry

**Status:** Accepted
**Date:** 2026-01-15

**Context:** In v0.2, `FieldManager` held field specs in a separate registry. Steps referenced fields by name only (`fields=["market_size", "competition"]`), and field definitions (prompts, types, enums) lived elsewhere. This created a split-brain problem: the step that *produces* a field didn't know *how* it should be produced. Changing a field prompt meant editing a separate file, then verifying the step still referenced it correctly. Mismatches were common and silent.

**Decision:** LLMStep accepts inline field specs: `fields={"market_size": "Estimate TAM"}` (shorthand) or full 7-key dicts with `prompt`, `type`, `format`, `enum`, `examples`, `bad_examples`, `default`. Each step declares what it produces AND how. `load_fields()` remains as a utility for teams where non-developers manage field prompts in CSV, but it's not in the core path — it just returns a dict that gets passed to LLMStep. FieldManager was removed from the public API.

**Alternatives considered:**
- **Central field registry (FieldManager)** — Scattered definition: field prompt in one place, step usage in another. Hard to track which step produces which field. Silent mismatches.
- **Field specs in config** — Config should be serializable and reusable across environments. Field specs contain prompt engineering details that are code, not configuration.
- **Separate YAML/JSON field definition files** — Introduces another file format. CSV already serves the "non-developer manages prompts" use case via `load_fields()`.
