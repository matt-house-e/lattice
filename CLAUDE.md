# Lattice - Enrichment Pipeline Engine

## What Lattice Is

A programmatic enrichment engine. The gap between Instructor (single LLM call) and Clay (full SaaS platform). Users define a pipeline of composable steps, point it at a DataFrame, and get structured results. Lattice handles orchestration: column-oriented batching, step dependencies, Pydantic validation, retries, checkpointing, and async concurrency.

## Architecture (v0.4)

**Column-oriented, composable steps:**
```
[All rows] → Step 1 (function/API)   → batch complete
[All rows] → Step 2 (LLM classify)   → batch complete
[All rows] → Step 3 (LLM synthesize) → batch complete
```

NOT row-oriented (that was v0.2). Each step runs across ALL rows before the next step starts. Independent steps run in parallel. Steps see outputs from their declared dependencies.

### Key Design Decisions

- **Pipeline is the primary API**: `pipeline.run(df)` is the ONE way to use Lattice. No Enricher in the public API — it's an internal runner.
- **Fields live on steps**: LLMStep accepts inline field specs. No separate field definition file required. Field spec keys: `prompt`, `type`, `format`, `enum`, `examples`, `bad_examples`, `default` (all optional except `prompt`).
- **Step protocol**: Async-only `run()` method. No sync/async duplication.
- **Provider-agnostic via LLMClient protocol**: OpenAI is the default (zero config). Anthropic and Google ship as optional extras (`pip install lattice[anthropic]`, `lattice[google]`). Any provider works via the `LLMClient` protocol (~30-line adapter). No litellm, no LangChain.
- **FunctionStep is the escape hatch**: Any external data source (APIs, web search, databases) is a FunctionStep. No built-in provider steps (no WebSearchStep).
- **Step data**: `dict[str, Any]` not `pd.Series`. Steps are pure, no pandas.
- **Internal fields**: `__` prefix (e.g. `__web_context`) for inter-step data, filtered from output.
- **No eval tooling**: Lattice exposes data (cost, errors, usage). Users run their own evals. Lifecycle hooks let observability tools plug in without being dependencies.
- **Minimal dependencies**: Base install: `openai`, `pydantic`, `pandas`, `tqdm`, `python-dotenv`. Optional: `anthropic`, `google-genai`. Never add heavy transitive deps (no litellm, no langfuse).
- **Caching via SQLite (`sqlite3` stdlib)**: Input-hash per-step-per-row cache. Cache key includes step name, row data, prior results, field specs, model, and temperature — changing a prompt auto-invalidates. WAL mode for concurrent access. `cache=False` per-step bypass. `FunctionStep(..., cache_version="v1")` for function caching. Cache stats flow into `PipelineResult.cost`.
- **Checkpoint + cache are separate concerns**: Checkpoint = step-level crash recovery (fast single-file resume). Cache = row-level input-hash deduplication (across runs). `checkpoint_interval=100` saves partial step progress every N rows for large datasets.
- **`list[dict]` input accepted**: `Pipeline.run()` takes `pd.DataFrame | list[dict]`. Internals already work on dicts. Serves server contexts, test code, and Polars users (`.to_dicts()`).
- **pandas stays as base dependency**: 77% of data practitioners use pandas. Lattice's users (enrichment workflows, Jupyter, CSV origins) are pandas users. No native Polars support — `list[dict]` covers the gap. Revisit post-launch.

### Public API

```python
from lattice import Pipeline, LLMStep, FunctionStep, EnrichmentConfig

# Primary: OpenAI (default, zero config)
pipeline = Pipeline([
    LLMStep("analyze", fields={
        "market_size": "Estimate TAM in billions USD",               # shorthand (prompt only)
        "competition": {                                              # full spec
            "prompt": "Rate competition level with key competitors",
            "enum": ["Low", "Medium", "High"],
            "examples": ["High - Competes with AWS, Google Cloud"],
        },
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
| 2 | Resilience + API redesign: error handling, retries, progress, cost, Pipeline.run(df), fields on steps | COMPLETE |
| 3 | Field spec + dynamic prompt (#33): 7-key field spec validation, dynamic prompt builder (markdown+XML), `default` enforcement, model default → gpt-4.1-mini, CSV loader update | COMPLETE |
| 4 | Caching + checkpoint enhancement (#17): SQLite input-hash cache, per-step-per-row, TTL expiry, cache stats, `checkpoint_interval`, `list[dict]` input | COMPLETE |
| 5A | Ship (#18): Working examples, README rewrite, PyPI publish (`lattice-enrichment`) | NOT STARTED |
| 5B | Power user features: Lifecycle hooks (#30), three-tier prompt customization (#34), web search utility (#35), CLI | NOT STARTED |

### Backlog Triage (Feb 2026)
- **#13 (Eval suite)** — Closed as won't-fix. Conflicts with design principle: evals are user-level, not library-level.
- **#11 (Streaming)** — Kept, needs re-scope for v0.3 column-oriented model. May merge into #30 (hooks).
- **#12 (Provenance)** — Kept, deprioritized. Partially addressed by web search two-step pattern. Post-launch differentiator.

## Design Principles

- **One way to do it.** `pipeline.run(df)` is the primary API. Power users get `pipeline.runner()`. No other entry points.
- **Fields live on steps.** LLMStep declares what it produces AND how. 7 field spec keys: `prompt` (required), `type`, `format`, `enum`, `examples`, `bad_examples`, `default`. No separate field registry.
- **No unnecessary dependencies.** Base install is 5 deps. Anthropic/Google are optional extras. Never add litellm, langfuse, or other heavy transitive deps.
- **FunctionStep is the escape hatch.** Any external data source is a FunctionStep. No built-in provider steps.
- **Evals are user-level, not library-level.** Lattice exposes inputs, outputs, cost, and errors. Users evaluate correctness in their own domain.
- **CSV is a utility, not a dependency.** `load_fields()` loads prompts from CSV for teams where non-devs manage fields. It's not in the core path.
- **Prompt engineering follows OpenAI cookbook.** Markdown headers for sections, XML tags for data boundaries. Dynamic prompt builder only includes keys actually used across fields. JSON in prompts is avoided (performs poorly per OpenAI's testing).

## Scale Limits & Sweet Spot

**Lattice's sweet spot: 100 to 50,000 rows.** Primary use case: 1,000-10,000 rows. This is the "too many for manual/Instructor, too few to justify big data infrastructure" zone.

| Rows | Memory | Time (3 steps, 10 workers) | Cost (gpt-4.1-mini) | Verdict |
|------|--------|---------------------------|---------------------|---------|
| 100 | Trivial | ~30s | ~$0.20 | Trivial |
| 1,000 | ~10-50MB | ~5 min | ~$2 | Bread and butter |
| 10,000 | ~100-500MB | ~50 min | ~$20 | Core use case, caching essential |
| 50,000 | ~500MB-2GB | ~4 hours (50 workers: ~50 min) | ~$100 | Upper bound, needs Tier 2+ rate limits |
| 100,000 | ~1-5GB | ~8 hours (200 workers: ~25 min) | ~$200 | Feasible but pushing it |
| 1,000,000 | ~10-50GB | Prohibitive single-process | ~$2,000 | Outgrown Lattice |

**When users outgrow Lattice:**
- **>50K-100K rows regularly**: Time is prohibitive without distributed processing (Spark, Ray, Prefect workers)
- **Real-time/streaming enrichment**: Lattice is batch-only — process arrives, wait for pipeline to finish
- **Multi-machine parallelism**: Need distributed workers, not single-process asyncio
- **Complex orchestration**: Conditional branching, loops, human-in-the-loop → Prefect/Dagster/Airflow
- **Data exceeds memory**: >10GB DataFrames → Dask, Polars lazy evaluation, Spark

**Lattice's constraints are single-process and in-memory pandas.** The bottleneck is almost always API rate limits and cost, not Lattice itself. With Tier 5 rate limits (200 workers), Lattice can process 50K rows across 3 steps in under an hour.

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
