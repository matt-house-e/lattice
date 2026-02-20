# 006: SQLite Cache with Input-Hash Keys

**Status:** Accepted
**Date:** 2026-02-01

**Context:** Through Phases 1-3, Lattice had no caching. Every run re-executed all LLM calls for all rows, even if nothing changed. During iterative prompt development — the primary workflow — this meant waiting minutes and spending dollars to test a one-word prompt tweak. The cache needed to auto-invalidate when inputs change (prompt, model, row data) without requiring manual cache clearing.

**Decision:** SQLite-backed cache using `sqlite3` (stdlib, zero dependencies). Cache key is SHA256 of canonical JSON containing: step name, row data, prior step results, field specs, model, and temperature. Changing any input — including a field prompt — changes the hash, so stale results are never served. WAL mode enables concurrent reads/writes. TTL expiry cleans old entries. Per-step bypass via `cache=False`. FunctionStep supports `cache_version` for invalidation when function logic changes (since function bodies aren't hashable).

**Alternatives considered:**
- **Filesystem JSON files** — Slower than SQLite (benchmarked ~35% slower per sqlite.org). Harder to manage TTL, no indexing, directory explosion with thousands of rows.
- **Redis** — External dependency. Overkill for local development. Not suitable for the "pip install and go" experience.
- **In-memory dict** — Lost on process restart. Useless for iterative prompt development across runs, which is the primary use case.
- **Content-addressable with manual invalidation** — Requires users to clear cache when prompts change. Error-prone, bad DX. Input-hash auto-invalidation is strictly better.
