# 005: FunctionStep as Universal Escape Hatch

**Status:** Accepted
**Date:** 2026-01-10

**Context:** Enrichment pipelines need external data: web search results, API calls, database lookups, file reads. The framework could ship built-in steps for common sources (WebSearchStep, APIStep, DatabaseStep) or provide a generic mechanism for users to bring any data source. Built-in steps would mean maintaining provider SDKs, handling authentication patterns, and shipping dependencies for services most users don't need.

**Decision:** FunctionStep wraps any async or sync callable. No built-in provider steps exist. Web search is a utility factory (`web_search()` returning an async callable), not a step type. The canonical pattern for external data is two steps: `FunctionStep` fetches data into an internal field (`__web_context`), then `LLMStep` analyzes it. Users bring their own provider SDKs, API keys, and error handling. FunctionStep supports `cache_version` for cache invalidation when function logic changes.

**Alternatives considered:**
- **Built-in WebSearchStep** — Couples Lattice to one search provider (originally Tavily was considered). Maintenance burden grows with each provider. Users still need FunctionStep for everything else.
- **Built-in API client steps** — Would need to support hundreds of APIs. Impossible to maintain, and each adds dependencies.
- **Plugin system for provider steps** — Adds framework complexity (plugin discovery, registration, versioning) for a problem FunctionStep already solves. Over-engineering.
- **`web_search()` as a step type** — Makes it a special case rather than a composable callable. As a factory returning an async function, it slots into FunctionStep naturally and can be composed with other functions.
