# Pipeline Architecture Design

> **Status**: Phases 1-2 COMPLETE. Phases 3-4 pending.
> **Version**: v0.3
> **Date**: February 2026

## Overview

Lattice v0.3 replaces the monolithic chain model with a composable pipeline of steps. Each step runs across all rows (column-oriented) before the next step starts. Steps declare dependencies; independent steps run in parallel.

## Step Protocol

```python
# lattice/steps/base.py

@dataclass(frozen=True)
class StepContext:
    """Immutable context passed to each step."""
    row: dict[str, Any]                # Original row data (converted from pd.Series at boundary)
    fields: dict[str, dict[str, Any]]  # Field specs for THIS step only (sliced by Pipeline)
    prior_results: dict[str, Any]      # Merged outputs from dependency steps
    config: Any = None                 # EnrichmentConfig

@dataclass
class StepResult:
    """Output from a single step execution."""
    values: dict[str, Any]              # Field name → produced value
    usage: Optional[UsageInfo] = None   # Token usage (LLM steps)
    metadata: dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class Step(Protocol):
    """Protocol all steps must satisfy. Async-only to eliminate duplication."""
    name: str
    fields: list[str]          # Field names this step produces
    depends_on: list[str]      # Step names this depends on

    async def run(self, ctx: StepContext) -> StepResult: ...
```

**Why async-only**: The v0.2 codebase had ~400 lines of sync/async duplication. Writing `run()` as async-only and providing a sync wrapper at the Enricher level eliminates this entirely.

**Why Protocol, not ABC**: Allows duck typing. Users can implement steps as plain classes without inheriting anything.

## Pipeline

```python
# lattice/pipeline/pipeline.py

class Pipeline:
    def __init__(self, steps: list[Step]): ...
    async def execute(self, rows: list[dict], all_fields: dict, config) -> list[dict]: ...
```

### Construction
- Validates: no duplicate step names, no missing dependencies, no cycles
- Builds `_step_map: dict[str, Step]`
- Computes `_execution_levels: list[list[str]]` via topological sort (Kahn's algorithm)

### Execution (column-oriented)
```
For each level in execution_levels:
    For each step in level (parallel via asyncio.gather):
        For each row (concurrent with semaphore):
            Build StepContext (row data, step's field specs, prior results from deps)
            result = await step.run(ctx)
            Store result
    Merge step results into accumulated results
Return accumulated results for all rows
```

Each level runs sequentially. Steps within a level run in parallel. Rows within a step run concurrently (bounded by semaphore from config.max_workers).

### Internal fields
Fields prefixed with `__` (e.g. `__web_context`) are used for inter-step communication. The Enricher filters them from the final DataFrame output.

## Built-in Steps

### LLMStep (`lattice/steps/llm.py`)

```python
class LLMStep:
    def __init__(
        self,
        name: str,
        fields: list[str] | dict[str, str | dict],  # Phase 2: inline specs
        depends_on: list[str] = None,
        model: str = "gpt-4.1-nano",
        temperature: float = None,    # Falls back to config
        max_tokens: int = None,       # Falls back to config
        system_prompt: str = None,    # Falls back to built-in enrichment prompt
        api_key: str = None,          # Falls back to OPENAI_API_KEY env
        base_url: str = None,         # Phase 2: OpenAI-compatible endpoints
        client: LLMClient = None,     # Phase 2: any LLMClient protocol adapter
        schema: Type[BaseModel] = EnrichmentResult,
        max_retries: int = 2,
    ): ...

    async def run(self, ctx: StepContext) -> StepResult: ...
```

**Fields parameter (Phase 2):**
- `list[str]` — field names only, specs come from external source (backward compat)
- `dict[str, str]` — shorthand: `{"market_size": "Estimate TAM"}` → prompt only
- `dict[str, dict]` — full spec: `{"market_size": {"prompt": "...", "type": "String", "instructions": "...", "examples": [...]}}`

When fields is a dict, LLMStep uses specs directly in `_build_system_message()` — no FieldManager needed.

- Uses `LLMClient` protocol internally (not `openai.AsyncOpenAI` directly)
- `OpenAIClient` is the default adapter (covers OpenAI + all compatible providers via `base_url`)
- `AnthropicClient` and `GoogleClient` ship as optional extras
- `client` parameter accepts any `LLMClient`-compatible adapter
- `base_url` shortcut creates `OpenAIClient(base_url=...)` for OpenAI-compatible providers
- Lazy client initialization (no import-time API key check)
- Builds system message with: system prompt + row data + field specs + prior_results + JSON schema
- Calls with `response_format={"type": "json_object"}`
- Validates response with Pydantic `model_validate()`
- On validation error: appends error to conversation, retries (sends error back to LLM)
- Returns `StepResult` with values filtered to only this step's declared fields

### FunctionStep (`lattice/steps/function.py`)

```python
class FunctionStep:
    def __init__(
        self,
        name: str,
        fn: Callable,          # (StepContext) -> dict[str, Any]
        fields: list[str],
        depends_on: list[str] = None,
    ): ...
```

- Wraps any sync or async callable
- Sync functions run via `run_in_executor`
- The escape hatch for any data source: APIs, databases, custom logic

### LLMClient Protocol (`lattice/steps/providers/base.py`) — Phase 2

```python
class LLMClient(Protocol):
    """What LLMStep needs from any LLM provider."""
    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict | None = None,
    ) -> LLMResponse: ...

@dataclass
class LLMResponse:
    content: str
    usage: UsageInfo | None = None
```

**Shipped adapters:**
- `OpenAIClient` (~30 lines) — default. Covers OpenAI, DeepSeek, Groq, Together, Fireworks, Ollama, vLLM, Mistral, LM Studio via `base_url`.
- `AnthropicClient` (~30 lines) — optional extra: `pip install lattice[anthropic]`
- `GoogleClient` (~30 lines) — optional extra: `pip install lattice[google]`

**Import:** `from lattice.providers import OpenAIClient, AnthropicClient, GoogleClient`

### No Built-in WebSearchStep

**Decision (Feb 2026): WebSearchStep was scrapped.** FunctionStep already handles any data source — web search, APIs, databases. Users bring their own search provider:

```python
async def web_search(ctx):
    results = await my_search_provider(ctx.row["company_name"])
    return {"__web_context": results}

Pipeline([
    FunctionStep("search", fn=web_search, fields=["__web_context"]),
    LLMStep("analyze", fields=["market_size"], depends_on=["search"]),
])
```

No Tavily dependency. No provider abstraction. No "waterfall resolution." Users write a function; Lattice orchestrates it.

## Primary API: Pipeline.run()

**Phase 2 redesign:** Pipeline becomes the primary public interface. Enricher becomes an internal runner.

```python
# lattice/pipeline/pipeline.py

class Pipeline:
    def __init__(self, steps: list[Step]): ...

    # Primary entry points (Phase 2)
    def run(self, df: pd.DataFrame, config: EnrichmentConfig = None) -> pd.DataFrame: ...
    async def run_async(self, df: pd.DataFrame, config: EnrichmentConfig = None) -> pd.DataFrame: ...

    # Power user: reusable runner with config
    def runner(self, config: EnrichmentConfig = None) -> Enricher: ...

    # Core execution (Phase 1, unchanged)
    async def execute(self, rows, all_fields, config, ...) -> list[dict]: ...
```

### `run()` implementation
1. Collect field specs from steps (LLMStep provides specs, FunctionStep provides names only)
2. Create internal Enricher with config
3. Convert DataFrame rows to `list[dict]`
4. Call `self.execute(rows, fields, config)`
5. Write results back to DataFrame (filtering `__` prefixed internal fields)
6. Return enriched DataFrame

### Enricher (internal runner)

```python
# lattice/core/enricher.py — internal, not in public API

class Enricher:
    def __init__(self, pipeline: Pipeline, config: EnrichmentConfig = None): ...
    def run(self, df: pd.DataFrame) -> pd.DataFrame: ...
    async def run_async(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

Enricher handles: DataFrame conversion, checkpointing, `__` field filtering, sync/async wrapping. Created via `pipeline.runner(config=...)` for repeated execution with the same config.

No FieldManager parameter. No category parameter. Field specs come from steps.

## Dependencies

### Remove
- `langchain>=0.3.0`
- `langchain-core>=0.3.0`
- `langchain-openai>=0.3.0`
- `tenacity>=9.0.0`

### Add
- `openai>=1.0.0`
- `pydantic>=2.0.0`

### Keep
- `pandas>=2.0.0`
- `tqdm>=4.65.0`
- `python-dotenv>=1.0.0`

### Optional Extras (Phase 2)
- `anthropic>=1.0.0` — `pip install lattice[anthropic]`
- `google-genai>=1.0.0` — `pip install lattice[google]`
- `pip install lattice[all]` — both

## Files Ported Unchanged

- `lattice/schemas/__init__.py`, `base.py`, `enrichment.py`, `structured.py`
- `lattice/core/config.py`
- `lattice/core/exceptions.py` (extend with `StepError`, `PipelineError`)
- `lattice/data/fields.py` (fix examples column bug — see below)
- `lattice/utils/logger.py`

**NOTE: `lattice/core/checkpoint.py` must be REDESIGNED, not ported.** Column-oriented execution changes checkpoint granularity from per-row to per-step. After step 1 completes for all rows, save step 1 results. If step 2 fails, resume from step 2 (step 1 results are preserved). The current `CheckpointManager` stores `last_processed_idx` (a row index) which doesn't map to the new model.

New checkpoint model:
- Save after each step completes for all rows
- Store: `{completed_steps: ["step1", "step2"], step_results: {...}, metadata: {...}}`
- On resume: skip completed steps, start from next step with saved results as prior_results
- Still save to CSV + JSON (same file format, different content structure)

## Field Routing Validation

Pipeline validates at construction time:
- No duplicate step names, no missing dependencies, no cycles (already done in Phase 1)
- No two steps produce the same non-`__` field

Optional validation at `run()` time:
- `pipeline.run(df, expected_fields=["market_size", "competition"])` validates coverage
- Missing fields → `FieldValidationError` (fail fast)
- No category concept — just a flat list of expected field names

## System Prompt

The LLMStep default system prompt must be ported from the current `LLMChain._create_default_prompt()` in `lattice/chains/llm.py`. This is ~50 lines of carefully crafted prompt engineering that instructs the LLM to act as a "structured data enrichment engine." It significantly affects output quality.

## StructuredResult[T] Status

`StructuredResult[T]` from `lattice/schemas/structured.py` is kept in the schemas package but is **not used in the core pipeline flow**. `LLMStep.run()` returns `StepResult` (which has its own `usage` field). `StructuredResult` remains available for users who want to use schemas standalone.

## Known Bug Fix: fields.py

Line 113 uses `df.columns[5:]` to grab example columns by position. With V2 CSV format, this grabs V2-specific columns (Sources, Good_Example, etc.) as "examples" even though they're handled separately. Fix:

```python
KNOWN_COLUMNS = {'Category', 'Field', 'Prompt', 'Data_Type', 'Instructions',
                 'Output_Format', 'Quality_Rules', 'Sources',
                 'Good_Example', 'Bad_Example', 'Fallback'}
examples = []
for col in df.columns:
    if col not in KNOWN_COLUMNS and pd.notna(row[col]):
        examples.append(str(row[col]))
```

## Files Deleted

- `lattice/chains/` (entire directory — replaced by `lattice/steps/`)
- `lattice/vector_store/` (entire directory — dead code, broken imports)
- `lattice/core/processors.py` (replaced by `lattice/pipeline/`)

## Usage Examples

### Simple: One LLM step (primary API)
```python
from lattice import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze", fields={
        "market_size": "Estimate the total addressable market in billions USD",
        "competition": "Rate as Low/Medium/High with key competitors",
        "growth_potential": "Evaluate growth potential with reasoning",
    })
])
result = pipeline.run(df)
```

### Multi-step with dependencies
```python
from lattice import Pipeline, LLMStep, FunctionStep

async def search_company(ctx):
    results = await my_search(ctx.row["name"])
    return {"__web_context": results}

pipeline = Pipeline([
    FunctionStep("search", fn=search_company, fields=["__web_context"]),
    LLMStep("market", fields={
        "market_size": "Estimate TAM using search results",
        "competition": "Rate competition level",
    }),
    # search + market run in parallel, then synthesis uses both
    LLMStep("synthesis", fields={
        "growth_potential": "Synthesize growth potential from market data and search results",
    }, depends_on=["search", "market"]),
])
result = pipeline.run(df)
```

### Custom function step
```python
def lookup_funding(ctx):
    company = ctx.row.get("company_name")
    return {"funding_amount": my_api_call(company)}

pipeline = Pipeline([
    FunctionStep("crunchbase", fn=lookup_funding, fields=["funding_amount"]),
    LLMStep("analysis", fields={
        "investment_thesis": "Write an investment thesis based on funding data",
    }, depends_on=["crunchbase"]),
])
result = pipeline.run(df)
```

### Power user: reusable runner
```python
from lattice import EnrichmentConfig

runner = pipeline.runner(config=EnrichmentConfig(
    max_workers=10,
    enable_checkpointing=True,
))
result = await runner.run_async(df)
```

### Team with CSV field definitions
```python
from lattice.data import load_fields

fields = load_fields("fields.csv", category="business_analysis")
pipeline = Pipeline([LLMStep("analyze", fields=fields)])
result = pipeline.run(df)
```

---

## Phase 2: Resilience + API Redesign (Epic #27)

Phase 1 built the happy path. Phase 2 makes it work for real workloads and simplifies the public API.

### API Redesign (#29)

The biggest change in Phase 2. Simplifies from 5 concepts to 2.

- `Pipeline.run(df)` becomes the primary entry point
- Fields defined inline on LLMStep (dict of field → prompt)
- Enricher becomes internal runner (accessed via `pipeline.runner()`)
- FieldManager demoted to `load_fields()` utility in `lattice.data`
- No categories in core — flat field lists
- Remove `EnrichmentSpec` (unused), clean up dead FieldManager methods

### Per-Row Error Handling (#23)

**Problem:** `asyncio.gather()` in `_execute_step` propagates one row's exception and kills all rows.

**Fix:** Use `return_exceptions=True` or equivalent. Failed rows produce error sentinels. Pipeline returns partial results + error report. Configurable: raise-on-any-failure vs. collect-and-continue.

### API-Level Retry with Backoff (#24)

**Problem:** LLMStep retries JSON parse errors (smart) but crashes on 429s, 500s, timeouts.

**Fix:** Catch `openai.RateLimitError`, `openai.APIError`, `openai.APITimeoutError`. Exponential backoff with jitter. Respect `Retry-After` header. Separate retry budget from parse-error retries.

### Progress Reporting (#20)

**Fix:** Wire tqdm into `Pipeline.execute()`. Step-level progress bar. Wire `progress_callback` from config.

### Cost Aggregation (#25)

**Problem:** `StepResult.usage` is discarded — `pipeline.py:203` only keeps `.values`.

**Fix:** Collect usage across rows and steps. Pipeline exposes cost summary after completion.

### Config Cleanup (#19) + Dead Code Removal (#26)

Remove or wire unused config fields. Remove dead exception classes (`VectorStoreError`, `LLMError`). Fix factory presets.

### LLMStep Provider Flexibility (#28)

Add `base_url` and `client` parameters. No litellm dependency — users bring their own client.

## Phase 3: Caching (Epic #17)

Input-hash cache: `hash(step_name + row_data + field_specs)` → cached result. Filesystem backend (JSON). TTL-based expiry. Manual bypass. Per-step enable/disable.

## Phase 4: Polish

- Lifecycle hooks for observability (#30) — callbacks at pipeline/step/row boundaries
- Working examples with sample data
- CLI: `lattice run --csv data.csv --fields fields.csv`
- PyPI publish: `pip install lattice-enrichment`
- README rewrite with real examples

## Design Decisions Log

| Decision | Date | Rationale |
|----------|------|-----------|
| `Pipeline.run(df)` as primary API | Feb 2026 | One concept, one entry point. Enricher is internal detail, not public API. |
| Fields live on steps | Feb 2026 | Single source of truth. No separate field registry. LLMStep declares what AND how. |
| FieldManager → `load_fields()` utility | Feb 2026 | CSV loading is a convenience, not a core dependency. No categories in core. |
| Drop WebSearchStep | Feb 2026 | FunctionStep already handles any data source. No built-in provider steps. |
| Drop waterfall resolution | Feb 2026 | Source priority is user logic (domain-specific), not framework logic. |
| LLMClient protocol + shipped adapters | Feb 2026 | OpenAI default, Anthropic/Google as optional extras (~30 lines each). Covers top 3 providers + all OpenAI-compatible via base_url. |
| No litellm dependency | Feb 2026 | Too heavy (~30 transitive deps). Protocol + thin adapters achieves the same with zero required deps. |
| No langfuse/eval dependency | Feb 2026 | Evals are application-level. Lattice exposes hooks for observability tools to plug in. |
| Phase 2 = Resilience + API redesign | Feb 2026 | Foundation gaps + API simplification must happen before PyPI publish. |
