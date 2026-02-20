# 009: Structured Outputs Auto-Detect

**Status:** Accepted
**Date:** 2026-02-10

**Context:** OpenAI's `json_schema` + `strict: true` response format constrains token generation at the API level, eliminating most JSON parse/validation errors. However, not all providers support it: `base_url` providers (Ollama, Groq) may not, and the feature behaves differently across OpenAI, Anthropic, and Google. Requiring users to know which mode their provider supports creates a footgun. Always using the legacy `json_object` mode wastes OpenAI's stricter guarantees.

**Decision:** Auto-detect the right response format based on provider and field configuration. Native OpenAI client + dict field specs = `json_schema` + `strict: true` with a dynamically built Pydantic model from field specs. `base_url` providers, non-OpenAI clients, custom response schemas, or list-type fields fall back to `json_object`. Users can override with `structured_outputs=True` or `structured_outputs=False` on LLMStep. The dynamic model is built by `schema_builder.py` using Pydantic v2's `create_model()`, deriving JSON schema from the 7-key field spec.

**Alternatives considered:**
- **Always use json_object** — Misses OpenAI's stricter token-level guarantees. More retries needed for malformed JSON. Leaves performance on the table.
- **Always use json_schema** — Breaks on providers that don't support it (Ollama, older OpenAI-compatible endpoints). Silent failures or cryptic errors.
- **Require explicit structured_outputs=True/False** — Pushes provider knowledge onto users. "Do the right thing by default" is better DX. Power users who need control still have the override.
- **Users write Pydantic models per step** — Duplicates the field spec (prompt, type, enum already define the schema). Maintenance burden, easy to drift.
