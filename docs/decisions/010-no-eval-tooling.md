# 010: No Eval Tooling

**Status:** Accepted
**Date:** 2026-02-10

**Context:** Issue #13 proposed building an eval suite into Lattice: ground truth comparisons, LLM-as-judge scoring, accuracy metrics. This is a common feature in LLM frameworks (LangChain, DSPy). The question was whether Lattice should ship eval tools or leave evaluation to users.

**Decision:** Closed #13 as won't-fix. Lattice exposes data — `PipelineResult` contains outputs, cost, errors, usage, and timing. Lifecycle hooks (`on_row_complete`, `on_pipeline_end`) let observability tools plug in without being dependencies. Users run their own evals in their own domain. What constitutes "correct" is domain-specific: TAM estimates might be validated against expert judgment; lead scoring against closed deals; classification against labeled datasets. Lattice can't know the evaluation criteria, so it shouldn't pretend to.

**Alternatives considered:**
- **Build eval suite into Lattice** — Drags in heavy dependencies (eval frameworks, metrics libraries). Makes the library opinionated about what "correct" means. Maintenance burden for something users will customize anyway.
- **Ship eval connectors (LangChain evals adapter, langfuse integration)** — Creates dependency on external eval systems. Their versioning and API changes become Lattice's problem. Users who don't use those systems pay the dependency cost.
- **Lightweight eval utilities (accuracy(), precision())** — Still couples Lattice to an eval philosophy. Simple metrics are trivial for users to compute from PipelineResult. The value-add doesn't justify the coupling.
