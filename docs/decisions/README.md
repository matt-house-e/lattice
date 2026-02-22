# Architecture Decision Records

This directory contains lightweight ADRs for Accrue's major architectural decisions. Each record captures the context, decision, and rejected alternatives in a single paragraph-style format.

## Template

Each file: `docs/decisions/NNN-short-title.md`

```markdown
# NNN: Short Title

**Status:** Accepted | Superseded by NNN
**Date:** YYYY-MM-DD

**Context:** What problem or question prompted this decision?

**Decision:** What was decided and why?

**Alternatives considered:** What else was evaluated and why was it rejected?
```

## Process

New architectural decisions get an ADR at decision time. CLAUDE.md and MEMORY.md hold the *current* state; this directory holds the *history*.

## Index

| # | Title | Status |
|---|-------|--------|
| 001 | [Pipeline.run() as sole public API](001-pipeline-run-sole-api.md) | Accepted |
| 002 | [Fields live on steps](002-fields-on-steps.md) | Accepted |
| 003 | [Column-oriented execution](003-column-oriented-execution.md) | Accepted |
| 004 | [Provider-agnostic via LLMClient protocol](004-llmclient-protocol.md) | Accepted |
| 005 | [FunctionStep as universal escape hatch](005-functionstep-escape-hatch.md) | Accepted |
| 006 | [SQLite cache with input-hash keys](006-sqlite-input-hash-cache.md) | Accepted |
| 007 | [Cache and checkpoint as separate concerns](007-cache-checkpoint-separate.md) | Accepted |
| 008 | [Lifecycle hooks on run(), not config](008-hooks-on-run.md) | Accepted |
| 009 | [Structured outputs auto-detect](009-structured-outputs-auto-detect.md) | Accepted |
| 010 | [No eval tooling](010-no-eval-tooling.md) | Accepted |
| 011 | [Conditional step execution via run_if/skip_if](011-conditional-step-predicates.md) | Accepted |
| 012 | [Provider-level grounding via `grounding` on LLMStep](012-provider-grounding.md) | Accepted |
