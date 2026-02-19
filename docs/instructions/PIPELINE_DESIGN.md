# Pipeline Architecture Design

> **Status**: Approved, not yet implemented
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
        fields: list[str],
        depends_on: list[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = None,    # Falls back to config
        max_tokens: int = None,       # Falls back to config
        system_prompt: str = None,    # Falls back to built-in enrichment prompt
        api_key: str = None,          # Falls back to OPENAI_API_KEY env
        schema: Type[BaseModel] = EnrichmentResult,
        max_retries: int = 2,
    ): ...

    async def run(self, ctx: StepContext) -> StepResult: ...
```

- Uses `openai.AsyncOpenAI` directly (no LangChain)
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

### WebSearchStep (`lattice/steps/web_search.py`) — Phase 2

Not in Phase 1. Composition example for when it exists:
```python
Pipeline([
    WebSearchStep("search"),
    LLMStep("analyze", fields=["market_size"], depends_on=["search"]),
])
```

## Enricher

```python
# lattice/core/enricher.py

class Enricher:
    def __init__(self, pipeline: Pipeline, field_manager: FieldManager, config: EnrichmentConfig = None): ...
    def run(self, df: pd.DataFrame, category: str, ...) -> pd.DataFrame: ...           # Sync wrapper
    async def run_async(self, df: pd.DataFrame, category: str, ...) -> pd.DataFrame: ... # Real impl
```

### `run()` sync wrapper
```python
def run(self, df, category, **kwargs):
    try:
        loop = asyncio.get_running_loop()
        # Already in async context (Jupyter, FastAPI) - warn and use nest_asyncio or thread
    except RuntimeError:
        return asyncio.run(self.run_async(df, category, **kwargs))
```

### `run_async()` implementation
1. Validate category exists in FieldManager
2. Get field specs via `field_manager.get_specs_as_dict(category)`
3. Load checkpoint if available
4. Convert DataFrame rows to `list[dict]`
5. Call `pipeline.execute(rows, fields, config)`
6. Write results back to DataFrame (filtering `__` prefixed internal fields)
7. Save checkpoint at intervals
8. Return enriched DataFrame

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
- `tavily-python>=0.3.0` (for Phase 2)

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

The Enricher must validate at `run()` time:
- Every field in the requested category has exactly one Pipeline step producing it
- No step produces fields not in the category (warning, not error)
- Missing fields → `FieldValidationError` (fail fast, don't silently drop fields)

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

### Simple: One LLM step
```python
from lattice import Enricher, Pipeline, LLMStep, FieldManager

pipeline = Pipeline([
    LLMStep("analyze", fields=["market_size", "competition_level", "growth_potential"])
])
enricher = Enricher(pipeline=pipeline, field_manager=FieldManager.from_csv("fields.csv"))
result = enricher.run(df, category="business_analysis")
```

### Multi-step with dependencies (Phase 2+)
```python
pipeline = Pipeline([
    WebSearchStep("search"),
    LLMStep("market", fields=["market_size", "competition_level"]),
    # search + market run in parallel, then synthesis uses both
    LLMStep("synthesis", fields=["growth_potential"], depends_on=["search", "market"]),
])
```

### Custom function step
```python
def lookup_funding(ctx):
    company = ctx.row.get("company_name")
    return {"funding_amount": my_api_call(company)}

pipeline = Pipeline([
    FunctionStep("crunchbase", fn=lookup_funding, fields=["funding_amount"]),
    LLMStep("analysis", fields=["investment_thesis"], depends_on=["crunchbase"]),
])
```
