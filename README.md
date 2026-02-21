<p align="center">
  <strong>Lattice</strong><br>
  <em>Enrich 10,000 rows with structured LLM outputs. Not one row. Not a million. The messy middle.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/lattice-enrichment/"><img src="https://img.shields.io/pypi/v/lattice-enrichment?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/lattice-enrichment/"><img src="https://img.shields.io/pypi/pyversions/lattice-enrichment" alt="Python"></a>
  <a href="https://github.com/lattice-team/lattice-enrichment/blob/main/LICENSE"><img src="https://img.shields.io/github/license/lattice-team/lattice-enrichment" alt="License"></a>
</p>

---

**Lattice is a programmatic enrichment engine.** Define a pipeline of composable steps, point it at a DataFrame, get structured results back. Lattice handles the orchestration you'd otherwise build yourself: column-oriented batching, step dependencies, Pydantic validation, retries, caching, checkpointing, and async concurrency.

**The gap between [Instructor](https://github.com/instructor-ai/instructor) and [Clay](https://www.clay.com/).** Instructor is great for a single LLM call. Clay is a full SaaS platform. Lattice is the missing middle: a Python library for running structured enrichment pipelines across hundreds to tens of thousands of rows. Version-control your enrichment logic, iterate on prompts with cached intermediate results, and pay API costs instead of SaaS markups.

```python
from lattice import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze", fields={
        "market_size": "Estimate total addressable market in billions USD",
        "competition": {
            "prompt": "Rate competitive intensity with key competitors",
            "enum": ["Low", "Medium", "High"],
            "examples": ["High - Competes with AWS, Google Cloud"],
        },
        "growth_potential": {
            "prompt": "Assess 5-year growth trajectory",
            "type": "String",
            "format": "X% CAGR - reasoning",
        },
    })
])

result = pipeline.run(df)  # DataFrame in, DataFrame out
print(result.data.head())
print(f"Tokens used: {result.cost.total_tokens:,}")
```

## Install

Requires Python 3.10+.

```bash
pip install lattice-enrichment
```

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

That's it. OpenAI is the default provider (zero config, [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) auto-enabled). Anthropic and Google are optional:

```bash
pip install lattice-enrichment[anthropic]  # Claude
pip install lattice-enrichment[google]     # Gemini
```

## Why Lattice

| | Instructor | Lattice | Clay |
|---|---|---|---|
| **Scope** | Single LLM call | Pipeline of steps across rows | Full SaaS platform |
| **Input** | One object | DataFrame / list[dict] | Spreadsheet UI |
| **Steps** | 1 | N (LLM, functions, APIs, web search) | N (drag-and-drop) |
| **Dependencies** | - | DAG with parallel execution | Sequential |
| **Caching** | - | SQLite per-step-per-row | Platform-managed |
| **Cost** | API costs | API costs | $$$$/month + API costs |
| **Version control** | Yes | Yes | No |

## Core Concepts

Lattice runs **column-oriented** enrichment. Each step processes ALL rows before the next step starts. Independent steps run in parallel.

```
[All rows] --> Step 1 (web search)    --> batch complete
[All rows] --> Step 2 (LLM classify)  --> batch complete
[All rows] --> Step 3 (LLM synthesize, depends on 1+2) --> batch complete
```

Three building blocks:

- **`Pipeline`** -- orchestrates a DAG of steps. `pipeline.run(df)` is the primary API.
- **`LLMStep`** -- calls an LLM with structured field specs. Validates with Pydantic, retries on parse errors.
- **`FunctionStep`** -- wraps any sync/async callable. The escape hatch for APIs, databases, web search, anything.

## Examples

### Multi-step pipeline with dependencies

```python
from lattice import Pipeline, FunctionStep, LLMStep, web_search

pipeline = Pipeline([
    # Step 1: Search the web for each company
    FunctionStep("research",
        fn=web_search("Research {company}: market position, competitors, recent news"),
        fields=["__web_context", "sources"],
    ),
    # Step 2: Analyze using search results + row data
    LLMStep("analyze",
        fields={
            "market_size": "Estimate TAM in billions USD",
            "competitors": {
                "prompt": "List top 3 competitors",
                "type": "List[String]",
            },
            "investment_thesis": "One-paragraph investment thesis",
        },
        depends_on=["research"],
    ),
])

result = pipeline.run(companies_df)
```

`web_search()` is a built-in utility wrapping OpenAI's Responses API. Template queries with `{field}` placeholders, automatic source extraction, graceful degradation on errors.

### Use any LLM provider

```python
# Anthropic (Claude)
from lattice.providers import AnthropicClient

LLMStep("analyze",
    fields={"summary": "Summarize this company's business model"},
    model="claude-sonnet-4-5-20250929",
    client=AnthropicClient(),
)

# Google (Gemini)
from lattice.providers import GoogleClient

LLMStep("analyze",
    fields={"summary": "Summarize this company's business model"},
    model="gemini-2.5-flash",
    client=GoogleClient(),
)

# Any OpenAI-compatible API (Ollama, Groq, Together, etc.)
LLMStep("analyze",
    fields={"summary": "Summarize this company's business model"},
    model="llama3",
    base_url="http://localhost:11434/v1",
)
```

Want a provider we don't ship? The `LLMClient` protocol is one async method:

```python
class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...
```

### Rich field specifications

Every field supports 7 spec keys for precise control over LLM output:

```python
LLMStep("classify", fields={
    # Shorthand: just a prompt string
    "market_size": "Estimate total addressable market",

    # Full spec: all 7 keys available
    "risk_level": {
        "prompt": "Assess overall investment risk",
        "type": "String",
        "enum": ["Low", "Medium", "High", "Very High"],
        "format": "Level - One sentence explanation",
        "examples": ["High - Pre-revenue with strong competition from incumbents"],
        "bad_examples": ["It's risky"],
        "default": "Unable to assess",
    },
})
```

### Caching and checkpointing

```python
from lattice import EnrichmentConfig

config = EnrichmentConfig(
    enable_caching=True,          # SQLite cache: skip redundant API calls
    cache_ttl=86400,              # 24-hour TTL
    enable_checkpointing=True,    # Resume from last completed step on crash
    checkpoint_interval=100,      # Save partial progress every 100 rows
    max_workers=30,               # Concurrent rows per step
)

result = pipeline.run(df, config=config)

# Cache key includes: step name, row data, field specs, model, temperature
# Change a prompt? Cache auto-invalidates. No stale results.
```

### Lifecycle hooks

Plug in observability without adding dependencies:

```python
from lattice import EnrichmentHooks

result = pipeline.run(df, hooks=EnrichmentHooks(
    on_pipeline_start=lambda e: print(f"Starting {len(e.step_names)} steps, {e.num_rows} rows"),
    on_row_complete=lambda e: print(f"  [{e.step_name}] Row {e.row_index}: {e.values}"),
    on_step_end=lambda e: print(f"  Step '{e.step_name}' done in {e.elapsed_seconds:.1f}s"),
    on_pipeline_end=lambda e: print(f"Done. {e.total_errors} errors, {e.elapsed_seconds:.1f}s total"),
))
```

Hooks are typed dataclasses. Sync and async callables both work. Hook errors are caught and logged -- they never crash your pipeline.

### Provider-level grounding

Let the LLM search the web natively -- no separate `FunctionStep` needed:

```python
# Basic: one flag enables grounding via the provider's native web search
LLMStep("research",
    fields={"summary": "Summarize recent news about this company"},
    grounding=True,
)

# With config: control domains, location, and provider-specific options
LLMStep("research",
    fields={"summary": "Summarize recent news"},
    grounding={
        "allowed_domains": ["crunchbase.com", "sec.gov"],
        "blocked_domains": ["reddit.com"],
        "user_location": {"country": "GB", "city": "London"},
        "max_searches": 3,
    },
)

# Works with any provider
from lattice.providers import AnthropicClient, GoogleClient

LLMStep("research", fields={...}, grounding=True, client=AnthropicClient())
LLMStep("research", fields={...}, grounding=True, client=GoogleClient())
```

Citations are normalized to `Citation(url, title, snippet)` across all providers and auto-injected as a `sources` field in the output. Customize with `sources_field="refs"` or disable with `sources_field=None`.

### Conditional step execution

Skip steps per-row based on data from prior steps:

```python
FunctionStep("enrich",
    fn=enrich_enterprise,
    fields=["analysis"],
    depends_on=["classify"],
    run_if=lambda row, prior: prior.get("tier") == "enterprise",
)

# Or skip rows that match a condition
LLMStep("deep_analysis",
    fields={"report": "Write a detailed analysis"},
    skip_if=lambda row, prior: prior.get("category") == "spam",
)
```

`run_if` and `skip_if` are mutually exclusive. Skipped rows get field spec `default` values (or `None`). Predicates run before cache checks -- skipped rows never hit the cache or the API.

### Custom function steps

Any external data source is a `FunctionStep`:

```python
async def fetch_funding(ctx):
    """Call Crunchbase API for funding data."""
    company = ctx.row["company"]
    resp = await httpx.AsyncClient().get(f"https://api.crunchbase.com/v4/entities/{company}")
    data = resp.json()
    return {
        "total_funding": data.get("funding_total", 0),
        "last_round": data.get("last_funding_type", "Unknown"),
    }

pipeline = Pipeline([
    FunctionStep("funding", fn=fetch_funding, fields=["total_funding", "last_round"]),
    LLMStep("thesis", fields={"investment_thesis": "..."}, depends_on=["funding"]),
])
```

Sync functions work too -- Lattice runs them via `run_in_executor` so they never block the event loop.

### Works with lists, not just DataFrames

```python
# list[dict] in, list[dict] out
result = pipeline.run([
    {"company": "Stripe", "sector": "Fintech"},
    {"company": "Notion", "sector": "Productivity"},
])
```

Handy for server contexts, test code, and Polars users (`.to_dicts()`).

## Configuration Presets

```python
from lattice import EnrichmentConfig

# Development: low concurrency, verbose, caching on
config = EnrichmentConfig.for_development()

# Production: high concurrency, checkpointing, caching
config = EnrichmentConfig.for_production()

# Server (FastAPI): no progress bars, high concurrency
config = EnrichmentConfig.for_server()
```

## Architecture

```
lattice/
├── steps/          # Step protocol + LLMStep, FunctionStep
│   └── providers/  # LLMClient protocol + OpenAI, Anthropic, Google adapters
├── pipeline/       # DAG resolution, column-oriented execution, run() entry point
├── schemas/        # Pydantic models (FieldSpec, CostSummary, UsageInfo)
├── core/           # Enricher (internal runner), config, cache, checkpoint, hooks
├── data/           # load_fields() CSV utility
└── utils/          # Logging, web_search
```

**Key design decisions:**

- **Column-oriented, not row-oriented.** Each step runs across all rows before the next starts. Independent steps at the same DAG level run in parallel.
- **Fields live on steps.** LLMStep declares what it produces AND how (7-key field specs). No separate field registry.
- **Provider-agnostic.** OpenAI is the default. Anthropic and Google ship as optional extras. Any provider works via the `LLMClient` protocol (~30 lines to implement).
- **FunctionStep is the escape hatch.** No built-in `WebSearchStep` or `DatabaseStep`. Any external data source is a FunctionStep with a plain callable.
- **Internal fields.** Prefix with `__` (e.g. `__web_context`) for inter-step data that gets filtered from the output.
- **Structured outputs auto-enabled.** Native OpenAI with dict fields gets `json_schema` + `strict: true` for constrained token generation. Auto-disabled for `base_url` or non-OpenAI providers that don't support it.

## Sweet Spot

Lattice is built for **100 to 50,000 rows.** The primary use case is 1,000-10,000 rows -- too many for manual work or single-call tools, too few to justify big data infrastructure.

| Rows | Time (3 steps, 10 workers) | Cost (gpt-4.1-mini) |
|------|---------------------------|---------------------|
| 100 | ~30s | ~$0.20 |
| 1,000 | ~5 min | ~$2 |
| 10,000 | ~50 min | ~$20 |
| 50,000 | ~50 min (50 workers) | ~$100 |

The bottleneck is almost always API rate limits and cost, not Lattice.

## API Reference

### `Pipeline(steps)`

Orchestrates a DAG of steps. Validates: no duplicate names, no missing dependencies, no cycles.

- `pipeline.run(data, config?, hooks?)` -- sync entry point. Returns `PipelineResult`.
- `await pipeline.run_async(data, config?, hooks?)` -- async entry point.
- `pipeline.runner(config?)` -- returns a reusable `Enricher` for repeated execution.
- `pipeline.clear_cache(step?, cache_dir?)` -- delete cached results.

### `LLMStep(name, fields, ...)`

Calls an LLM provider for structured enrichment. Fields accept `list[str]` or `dict[str, str | dict]`.

Key parameters: `model`, `temperature`, `system_prompt`, `system_prompt_header`, `client`, `base_url`, `max_retries`, `cache`, `structured_outputs`, `grounding`, `sources_field`, `run_if`, `skip_if`, `depends_on`.

### `FunctionStep(name, fn, fields, ...)`

Wraps any sync or async callable. Receives `StepContext`, returns `dict[str, Any]`.

Key parameters: `cache`, `cache_version`, `run_if`, `skip_if`, `depends_on`.

### `EnrichmentConfig`

Dataclass controlling concurrency, retries, caching, checkpointing, and progress display. All fields have sensible defaults.

### `PipelineResult`

- `.data` -- enriched DataFrame or list[dict] (matches input type)
- `.cost` -- `CostSummary` with per-step token usage and cache stats
- `.errors` -- list of `RowError` for failed rows
- `.success_rate` -- fraction of rows without errors
- `.has_errors` -- boolean

### Errors

All exceptions inherit from `EnrichmentError`:

```python
from lattice import EnrichmentError, FieldValidationError, StepError, PipelineError, RowError
```

### `GroundingConfig`

Pydantic model for grounding configuration. Fields: `allowed_domains`, `blocked_domains`, `user_location`, `max_searches`, `provider_kwargs`. Cross-provider fields are mapped to native format by each adapter; `provider_kwargs` is an escape hatch for provider-specific options.

```python
from lattice import GroundingConfig
```

## Contributing

```bash
git clone https://github.com/lattice-team/lattice-enrichment.git
cd lattice-enrichment
pip install -e ".[dev]"
pytest
```

## License

MIT
