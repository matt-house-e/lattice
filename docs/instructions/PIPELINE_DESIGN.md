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
    def run(self, df, config=None) -> PipelineResult: ...
    async def run_async(self, df, config=None) -> PipelineResult: ...
    def runner(self, config=None) -> Enricher: ...
    async def execute(self, rows, all_fields, config, ...) -> tuple[list[dict], list[RowError], CostSummary]: ...
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
        model: str = "gpt-4.1-mini",
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

**Fields parameter (Phase 2 → redesigned Phase 3):**
- `list[str]` — field names only, specs come from external source (backward compat)
- `dict[str, str]` — shorthand: `{"market_size": "Estimate TAM"}` → prompt only
- `dict[str, dict]` — full spec with 7 supported keys:

| Key | Type | Required | Purpose |
|-----|------|----------|---------|
| `prompt` | `str` | Yes | The extraction instruction |
| `type` | `str` | No | Data type: `String`, `Number`, `Boolean`, `Date`, `List[String]`, `JSON` (default: `String`) |
| `format` | `str` | No | Output format pattern (e.g. `"YYYY-MM-DD"`, `"$X.XB"`, `"X/10"`) |
| `enum` | `list[str]` | No | Constrained value list (e.g. `["Low", "Medium", "High"]`) |
| `examples` | `list[str]` | No | Good output examples showing expected style |
| `bad_examples` | `list[str]` | No | Anti-patterns to avoid |
| `default` | `Any` | No | Fallback value when data is insufficient (enforced in Python) |

Example:
```python
LLMStep("analyze", fields={
    "market_size": "Estimate TAM in billions USD",  # shorthand
    "risk_level": {                                  # full spec
        "prompt": "Assess investment risk based on market and competitive data",
        "type": "String",
        "enum": ["Low", "Medium", "High"],
        "default": "Unknown",
        "examples": ["High", "Medium"],
        "bad_examples": ["Moderately high", "3/5"],
    },
    "revenue": {
        "prompt": "Estimate annual revenue",
        "type": "Number",
        "format": "$X.XB",
    },
})
```

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

**Web search with citations (recommended pattern):**

OpenAI's web search + structured output in one call is unreliable (truncation bugs). The two-step pattern works:

```python
async def web_research(row):
    """FunctionStep: calls OpenAI Responses API with web search."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    response = await client.responses.create(
        model="gpt-4.1-mini",  # nano doesn't support web search
        tools=[{"type": "web_search", "search_context_size": "medium"}],
        input=f"Research {row['company']}: market position, competitors, recent news",
        include=["web_search_call.action.sources"],
    )
    sources = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                for ann in getattr(block, 'annotations', []):
                    if ann.type == "url_citation":
                        sources.append(ann.url)
    return {"__web_context": response.output_text, "sources": sources}

Pipeline([
    FunctionStep("research", fn=web_research, fields=["__web_context", "sources"]),
    LLMStep("analyze", fields={
        "market_size": {"prompt": "Estimate TAM based on the research context", "format": "$X.XB"},
        "competition": {"prompt": "Rate competition level", "enum": ["Low", "Medium", "High"]},
    }, depends_on=["research"]),
])
```

`__web_context` is internal (filtered from output). `sources` is visible — users get citation URLs per row. The LLMStep sees web context in `prior_results` via `depends_on`.

## Primary API: Pipeline.run()

**Phase 2 redesign:** Pipeline becomes the primary public interface. Enricher becomes an internal runner.

```python
# lattice/pipeline/pipeline.py

class Pipeline:
    def __init__(self, steps: list[Step]): ...

    # Primary entry points
    def run(self, df: pd.DataFrame, config: EnrichmentConfig = None) -> PipelineResult: ...
    async def run_async(self, df: pd.DataFrame, config: EnrichmentConfig = None) -> PipelineResult: ...

    # Power user: reusable runner with config
    def runner(self, config: EnrichmentConfig = None) -> Enricher: ...

    # Core execution (internal)
    async def execute(self, rows, all_fields, config, ...) -> tuple[list[dict], list[RowError], CostSummary]: ...
```

### `run()` implementation
1. Collect field specs from steps (LLMStep provides specs, FunctionStep provides names only)
2. Convert DataFrame rows to `list[dict]`
3. Call `self.execute(rows, fields, config)`
4. Write results back to DataFrame (filtering `__` prefixed internal fields)
5. Return `PipelineResult(data=df, cost=CostSummary, errors=list[RowError])`

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

## Checkpoint Model

Per-step checkpointing (implemented in Phase 1):
- Save after each step completes for all rows
- Store: `{completed_steps: ["step1", "step2"], step_results: {...}, metadata: {...}}`
- On resume: skip completed steps, start from next step with saved results as prior_results
- JSON file per pipeline run

## Field Routing Validation

Pipeline validates at construction time:
- No duplicate step names, no missing dependencies, no cycles (already done in Phase 1)
- No two steps produce the same non-`__` field

Optional validation at `run()` time:
- `pipeline.run(df, expected_fields=["market_size", "competition"])` validates coverage
- Missing fields → `FieldValidationError` (fail fast)
- No category concept — just a flat list of expected field names

## System Prompt (Redesign — Phase 3)

### Current (v0.3)
Static `DEFAULT_SYSTEM_PROMPT` in `lattice/steps/llm.py`. ~50 lines. Field specs injected as raw JSON via `json.dumps()`. Uses `response_format={"type": "json_object"}` (legacy).

### Planned: Dynamic Prompt Builder

Follows OpenAI GPT-4.1 cookbook: **markdown headers for sections, XML tags for data boundaries**. JSON in prompts performs poorly per OpenAI's long-context testing.

**Structure:**
```markdown
# Role
You are a structured data enrichment engine...

# Field Specification Keys
[DYNAMIC: only describe keys actually present across this step's fields]
- prompt: the extraction instruction
- type: expected output type
[only if any field uses enum:]
- enum: value MUST be one of these options verbatim
[only if any field uses format:]
- format: specific output format pattern
[etc.]

# Output Rules
- Return ONLY valid JSON. No prose, no code fences.
- Keys MUST be exactly the field names below.
[DYNAMIC: enum/default rules only if relevant]

<row_data>
{"company": "Acme Corp", "industry": "Cloud"}
</row_data>

<field_specifications>
<field name="market_size">
  <prompt>Estimate the total addressable market</prompt>
  <type>Number</type>
  <format>$X.XB</format>
</field>
<field name="risk_level">
  <prompt>Assess investment risk</prompt>
  <enum>Low, Medium, High</enum>
  <default>Unknown</default>
</field>
</field_specifications>

<prior_results>
[only if step has dependencies]
</prior_results>
```

**Key principles:**
- Static content at top (enables OpenAI prompt caching)
- Variable content (row data, field specs) at bottom
- Only include field spec key descriptions for keys actually used
- XML per-field blocks only include defined keys (no empty tags)
- Sandwich pattern: key constraints reiterated after data for long-context reliability

### Three-Tier Prompt Customization

1. **Default** — Dynamic prompt builder handles everything
2. **`system_prompt_header=`** — Inject domain context before field specs: `"You are analyzing B2B SaaS companies in the European market."`
3. **`system_prompt=`** — Full override (existing, power users own the entire prompt)

### Future: Structured Outputs Migration

Move from `response_format={"type": "json_object"}` (legacy) to `{"type": "json_schema", "strict": true}`. This enables:
- Enum fields enforced at grammar level (constrained decoding)
- Type enforcement (Number fields can't produce strings)
- Dynamically built Pydantic model from field specs at runtime
- Eliminates need for most parse retries

Research: `docs/research/prompt-engineering.md`

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

## Phase 2: Resilience + API Redesign (COMPLETE)

Merged in #32. All issues closed: #19, #20, #23, #24, #25, #26, #27, #28, #29.

### What was built

- **API Redesign (#29):** `Pipeline.run(df)` → `PipelineResult` as primary API. LLMStep accepts `fields={"name": "prompt"}` inline specs. Enricher demoted to internal runner via `pipeline.runner()`. `load_fields()` utility replaces FieldManager in public API. Removed `EnrichmentSpec`, `StructuredResult`, dead FieldManager methods.
- **Per-Row Error Handling (#23):** `asyncio.gather(*coros, return_exceptions=True)`. Failed rows produce `RowError` with sentinel values. Configurable `on_error="continue"|"raise"`.
- **API Retry with Backoff (#24):** Two-layer retry in LLMStep — outer loop for API errors (429/500/timeout) with exponential backoff + jitter + Retry-After header, inner loop for parse/validation errors fed back to LLM.
- **Progress Reporting (#20):** tqdm step-level progress bar wired to `config.enable_progress_bar`.
- **Cost Aggregation (#25):** `CostSummary` with per-step `StepUsage` (prompt/completion/total tokens, rows processed, model). Returned in `PipelineResult.cost`.
- **LLMClient Protocol (#28):** `LLMClient` protocol + `OpenAIClient` (default), `AnthropicClient`, `GoogleClient` adapters. `base_url` shortcut for OpenAI-compatible providers. `LLMAPIError` wrapper for provider-agnostic retry.
- **Dead Code Removal (#26) + Config Cleanup (#19):** Removed `VectorStoreError`, `LLMError`, `EnrichmentSpec`, `StructuredResult`. Config was already clean from Phase 1C.

## Phase 3: Field Spec Redesign + Dynamic Prompt (Epic #33)

The prompt engineering layer is the core of enrichment quality. This phase rebuilds it from research. **This is the highest-impact work remaining** — everything Lattice produces flows through the system prompt.

### Scope
1. **7-key field spec validation** — Enforce schema (`prompt`, `type`, `format`, `enum`, `examples`, `bad_examples`, `default`) with Pydantic validation on LLMStep construction. Reject unknown keys. `prompt` required, all others optional.
2. **Dynamic system prompt builder** — Markdown headers + XML data boundaries (OpenAI GPT-4.1 cookbook). Only describe keys actually present across this step's fields. Static content at top (enables OpenAI prompt caching), variable data at bottom.
3. **`default` enforcement in Python** — If field has `default` and LLM returns refusal language ("Unable to determine", "N/A", etc.), replace with default value post-extraction.
4. **Default model → `gpt-4.1-mini`** — Change hardcoded default. Nano stays available per-step.
5. **CSV loader update** — Map new 7-key spec from CSV columns. Concatenate legacy `Guidance` column into `prompt`. Support `examples`/`bad_examples` columns.

### Out of scope for Phase 3
- Structured Outputs migration (`json_schema` + `strict`) — requires dynamic Pydantic model generation, separate effort
- Regex validation on `format` — future enhancement
- `system_prompt_header` — Phase 5B (#34)

## Phase 4: Caching (Epic #17)

Input-hash cache for iterative development. Without caching, changing one field in a 3-step pipeline re-runs every API call for every row. This is the single biggest DX improvement.

### Scope
1. **Input-hash key** — `hash(step_name + row_data + field_specs + model)` → deterministic cache key
2. **Filesystem JSON backend** — Simple, zero deps, works everywhere
3. **TTL-based expiry** — `cache_ttl` already in `EnrichmentConfig` (unused), wire it up
4. **Per-step control** — `LLMStep(..., cache=False)` to bypass
5. **Pipeline-level control** — `EnrichmentConfig(enable_caching=True)` (already exists, wire it up)

**Note:** `enable_caching` and `cache_ttl` are already on `EnrichmentConfig` from Phase 1 — they're just not wired to anything. This phase makes them real.

## Phase 5A: Ship (Epic #18)

Minimum viable distribution. Get Lattice into users' hands.

- **Working examples** — 3-4 runnable scripts with sample data: simple enrichment, multi-step with deps, web search two-step pattern, Anthropic/Google provider usage
- **README rewrite** — Real examples, install instructions, quick start
- **PyPI publish** — `pip install lattice-enrichment`
- **#21 docs fix** — Update github-standards.md for v0.3 components (quick win)

## Phase 5B: Power User Features

- **Lifecycle hooks (#30)** — `EnrichmentHooks` with callbacks at pipeline/step/row boundaries. Enables Langfuse/Datadog/structlog integration without being dependencies.
- **Three-tier prompt customization (#34)** — `system_prompt_header=` injection. Depends on Phase 3's dynamic prompt builder.
- **Web search utility (#35)** — `web_search()` convenience function reducing boilerplate for the common two-step pattern.
- **CLI** — `lattice run --csv data.csv --fields fields.csv`

## Backlog Triage (Feb 2026)

| # | Issue | Decision | Reasoning |
|---|-------|----------|-----------|
| **#13** | Eval suite | **Closed (won't-fix)** | Conflicts with design principle: "Evals are user-level, not library-level." Lattice exposes data; users run their own evals. |
| **#11** | Streaming output | **Kept, re-scope** | Written for v0.2 row-oriented model. In column-oriented execution, "streaming" = step-level progress (already have tqdm) or row-level callbacks (that's #30 hooks). May merge into #30. |
| **#12** | Sources/provenance | **Kept, deprioritized** | Partially addressed by web search two-step pattern (`sources` column). Full provenance is a post-launch differentiator per TECHNICAL-VISION.md. |

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
| Default model → gpt-4.1-mini | Feb 2026 | Nano too limited: poor complex retrieval, no web search support, hallucination risk with structured outputs. Mini is "standout star," matches GPT-4o at 83% cheaper. Users can still set nano per-step for simple classification. |
| 7-key field spec (drop `instructions`) | Feb 2026 | Research-backed: `prompt`, `type`, `format`, `enum`, `examples`, `bad_examples`, `default`. Merged `instructions`/`guidance` into `prompt` — other keys already cover structured concerns. |
| Dynamic system prompt | Feb 2026 | Only describe field spec keys actually used. Saves tokens, avoids confusing gpt-4.1's literal instruction following with irrelevant sections. |
| Markdown headers + XML data boundaries | Feb 2026 | OpenAI GPT-4.1 cookbook recommendation. JSON in prompts "performed particularly poorly" in their long-context testing. |
| Web search = two-step FunctionStep → LLMStep | Feb 2026 | OpenAI web search + structured output in one call is broken (truncation). Two-step avoids this. Citations flow as visible `sources` column. |
| Three-tier prompt customization | Feb 2026 | Default (dynamic), `system_prompt_header=` (domain injection), `system_prompt=` (full override). Covers 99% of use cases without complexity. |
