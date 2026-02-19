# Lattice - Enrichment Pipeline Engine

## What Lattice Is

A programmatic enrichment engine. The gap between Instructor (single LLM call) and Clay (full SaaS platform). Users define a pipeline of composable steps, point it at a DataFrame, and get structured results. Lattice handles orchestration: column-oriented batching, step dependencies, Pydantic validation, retries, checkpointing, and async concurrency.

## Architecture (v0.3)

**Column-oriented, composable steps:**
```
[All rows] → Step 1 (web search)     → batch complete
[All rows] → Step 2 (LLM classify)   → batch complete
[All rows] → Step 3 (LLM synthesize) → batch complete
```

NOT row-oriented (that was v0.2). Each step runs across ALL rows before the next step starts. Independent steps run in parallel. Steps see outputs from their declared dependencies.

### Key Design Decisions

- **Step protocol**: Async-only `run()` method. No sync/async duplication.
- **Sync wrapper**: `Enricher.run()` calls `asyncio.run()` with event loop detection.
- **LLM SDK**: Direct `openai` SDK. No LangChain (dropped in v0.3).
- **Step data**: `dict[str, Any]` not `pd.Series`. Steps are pure, no pandas.
- **Internal fields**: `__` prefix (e.g. `__web_context`) for inter-step data, filtered from output.
- **Schemas**: Pydantic-first. `EnrichmentSpec` (input), `EnrichmentResult` (output), `StructuredResult[T]` (wrapper).

### Package Structure
```
lattice/
├── steps/          # Step protocol + built-in steps (LLMStep, FunctionStep, WebSearchStep)
├── pipeline/       # DAG resolution + column-oriented execution
├── schemas/        # Pydantic models (EnrichmentSpec, EnrichmentResult, StructuredResult)
├── core/           # Enricher, config, checkpoint, exceptions
├── data/           # FieldManager (CSV field definitions)
└── utils/          # Logging
```

### Public API
```python
from lattice import Enricher, Pipeline, LLMStep, FunctionStep, FieldManager, EnrichmentConfig
```

## Build Phases

Full design: `@docs/instructions/PIPELINE_DESIGN.md`

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Core pipeline engine: Step protocol, Pipeline, LLMStep, FunctionStep, Enricher rewrite | NOT STARTED |
| 2 | Web search + provider model: WebSearchStep, rate limiting, waterfall resolution | NOT STARTED |
| 3 | Caching + cost tracking: Input-hash cache, per-step cost reports | NOT STARTED |
| 4 | Polish: Docs, examples, CLI, PyPI publish | NOT STARTED |

### Phase 1 Work Chunks (each = one session)

| Chunk | Branch | What | Depends On |
|-------|--------|------|------------|
| 1A | `feature/pipeline-core` | Step protocol + FunctionStep + Pipeline + LLMStep + all unit tests | nothing |
| 1B | `feature/enricher-rewrite` | Enricher rewrite + checkpoint rethink (per-step, not per-row) + integration tests | 1A merged |
| 1C | `feature/v03-cleanup` | Delete old code, update deps/API, fix fields.py bug, update existing tests | 1B merged |

### Important: Checkpoint Must Be Rethought (not ported)
Column-oriented execution changes checkpoint granularity from per-row to per-step. After step 1 completes for all rows, save. Resume from last completed step. Current `CheckpointManager` has no concept of steps - this is redesigned in chunk 1B.

### Important: Field Routing Validation
Enricher must validate at `run()` time that every field in the requested category is covered by exactly one Pipeline step. Missing fields = error. Duplicate fields = error.

## Git Workflow

- **`main`** - Production-ready code
- **`feature/description`** - Feature branches

### Commit Format
```
type: Brief description

- Detail 1
- Detail 2

Co-Authored-By: Claude <noreply@anthropic.com>
```
Types: `feat`, `fix`, `docs`, `refactor`, `test`

### What Gets Committed
- Source code (`lattice/`), Tests (`tests/`), Examples (`examples/`), Docs (`.md`)
- Never: `data/`, `.env`, `.vscode/`, `.idea/`

## GitHub Issue Standards

**Always include labels.** Format: `[Type]: [Component] Description`

Labels: `type:{epic,story,task,bug,spike}`, `priority:{critical,high,medium,low}`, `component:{core,steps,pipeline,data,testing,docs,infra}`

See `docs/github-standards.md` for details.
