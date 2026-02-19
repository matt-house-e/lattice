# Lattice - Enrichment Pipeline Engine

## What Lattice Is

A programmatic enrichment engine. The gap between Instructor (single LLM call) and Clay (full SaaS platform). Users define a pipeline of composable steps, point it at a DataFrame, and get structured results. Lattice handles orchestration: column-oriented batching, step dependencies, Pydantic validation, retries, checkpointing, and async concurrency.

## Architecture (v0.3)

**Column-oriented, composable steps:**
```
[All rows] → Step 1 (function/API)   → batch complete
[All rows] → Step 2 (LLM classify)   → batch complete
[All rows] → Step 3 (LLM synthesize) → batch complete
```

NOT row-oriented (that was v0.2). Each step runs across ALL rows before the next step starts. Independent steps run in parallel. Steps see outputs from their declared dependencies.

### Key Design Decisions

- **Pipeline is the primary API**: `pipeline.run(df)` is the ONE way to use Lattice. No Enricher in the public API — it's an internal runner.
- **Fields live on steps**: LLMStep accepts inline field specs (prompt, type, instructions). No separate field definition file required.
- **Step protocol**: Async-only `run()` method. No sync/async duplication.
- **Provider-agnostic via LLMClient protocol**: OpenAI is the default (zero config). Anthropic and Google ship as optional extras (`pip install lattice[anthropic]`, `lattice[google]`). Any provider works via the `LLMClient` protocol (~30-line adapter). No litellm, no LangChain.
- **FunctionStep is the escape hatch**: Any external data source (APIs, web search, databases) is a FunctionStep. No built-in provider steps (no WebSearchStep).
- **Step data**: `dict[str, Any]` not `pd.Series`. Steps are pure, no pandas.
- **Internal fields**: `__` prefix (e.g. `__web_context`) for inter-step data, filtered from output.
- **No eval tooling**: Lattice exposes data (cost, errors, usage). Users run their own evals. Lifecycle hooks let observability tools plug in without being dependencies.
- **Minimal dependencies**: Base install: `openai`, `pydantic`, `pandas`, `tqdm`, `python-dotenv`. Optional: `anthropic`, `google-genai`. Never add heavy transitive deps (no litellm, no langfuse).

### Public API

```python
from lattice import Pipeline, LLMStep, FunctionStep, EnrichmentConfig

# Primary: OpenAI (default, zero config)
pipeline = Pipeline([
    LLMStep("analyze", fields={
        "market_size": "Estimate TAM in billions USD",
        "competition": "Rate Low/Medium/High with competitors",
    })
])
result = pipeline.run(df)

# Anthropic: pip install lattice[anthropic]
from lattice.providers import AnthropicClient
LLMStep("analyze", fields={...}, model="claude-sonnet-4-5-20250929", client=AnthropicClient())

# Google: pip install lattice[google]
from lattice.providers import GoogleClient
LLMStep("analyze", fields={...}, model="gemini-2.5-flash", client=GoogleClient())

# OpenAI-compatible (Ollama, Groq, DeepSeek, etc.): base_url shortcut
LLMStep("analyze", fields={...}, model="llama3", base_url="http://localhost:11434/v1")
```

### Package Structure
```
lattice/
├── steps/          # Step protocol + built-in steps (LLMStep, FunctionStep)
│   └── providers/  # LLMClient protocol + adapters (OpenAI, Anthropic, Google)
├── pipeline/       # DAG resolution + column-oriented execution + run() entry point
├── schemas/        # Pydantic models (EnrichmentResult)
├── core/           # Enricher (internal runner), config, checkpoint, exceptions
├── data/           # load_fields() utility for CSV field definitions
└── utils/          # Logging
```

## Build Phases

Full design: `@docs/instructions/PIPELINE_DESIGN.md`

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Core pipeline engine: Step protocol, Pipeline, LLMStep, FunctionStep, Enricher rewrite | COMPLETE |
| 2 | Resilience + API redesign: error handling, retries, progress, cost, Pipeline.run(df), fields on steps | NOT STARTED |
| 3 | Caching: Input-hash cache, filesystem backend, cache controls | NOT STARTED |
| 4 | Polish: Lifecycle hooks, docs, examples, CLI, PyPI publish | NOT STARTED |

## Design Principles

- **One way to do it.** `pipeline.run(df)` is the primary API. Power users get `pipeline.runner()`. No other entry points.
- **Fields live on steps.** LLMStep declares what it produces AND how (prompts, instructions). No separate field registry.
- **No unnecessary dependencies.** Base install is 5 deps. Anthropic/Google are optional extras. Never add litellm, langfuse, or other heavy transitive deps.
- **FunctionStep is the escape hatch.** Any external data source is a FunctionStep. No built-in provider steps.
- **Evals are user-level, not library-level.** Lattice exposes inputs, outputs, cost, and errors. Users evaluate correctness in their own domain.
- **CSV is a utility, not a dependency.** `load_fields()` loads prompts from CSV for teams where non-devs manage fields. It's not in the core path.

## Keeping Docs in Sync

**When making architectural decisions, design changes, or closing/opening build phases, update ALL of these:**

1. `CLAUDE.md` — This file. Architecture, phases, design decisions.
2. `docs/instructions/PIPELINE_DESIGN.md` — Detailed technical design.
3. `.notes/TECHNICAL-VISION.md` — Long-term vision and strategy.
4. GitHub issues — Close stale issues, update epics, create new issues.
5. Memory (`MEMORY.md`) — Current state, what's next, conventions.

**Never let these diverge.** A design decision discussed in conversation but not recorded in docs will be forgotten or contradicted. Update docs in the same session as the decision.

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
