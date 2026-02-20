# API Reference

Detailed documentation for Lattice's core components.

## Enricher

The main orchestrator for DataFrame enrichment. Validates field routing, manages per-step checkpoints, and provides sync/async DataFrame APIs.

```python
from lattice import Enricher, Pipeline, FieldManager

enricher = Enricher(
    pipeline=pipeline,
    field_manager=field_manager,
    config=config  # optional
)

# Synchronous (wraps asyncio.run)
result = enricher.run(df, category="business_analysis")

# Async
result = await enricher.run_async(df, category="business_analysis")
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | `Pipeline` | required | The step pipeline to execute |
| `field_manager` | `FieldManager` | required | Field definitions loaded from CSV |
| `config` | `EnrichmentConfig` | `EnrichmentConfig()` | Configuration options |

### Methods

| Method | Description |
|--------|-------------|
| `run(df, category, overwrite_fields=None)` | Sync entry point. Raises `RuntimeError` if called inside an async context. |
| `run_async(df, category, overwrite_fields=None)` | Async entry point. Validates fields, runs pipeline, writes results to DataFrame. |

### Field Routing Validation

At `run()` time, the Enricher validates:
- Every field in the requested category is produced by exactly one pipeline step
- No category field is produced by multiple steps (raises `FieldValidationError`)
- Extra step fields not in the category trigger a warning

---

## Pipeline

DAG-based execution engine. Resolves step dependencies via topological sort and executes level-by-level.

```python
from lattice import Pipeline, LLMStep, FunctionStep

pipeline = Pipeline([
    FunctionStep("search", fn=search_fn, fields=["__web_ctx"]),
    LLMStep("analyze", fields=["market_size"], depends_on=["search"]),
])
```

### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `steps` | `list[Step]` | Steps to execute. Validated for: no duplicate names, no missing dependencies, no cycles. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `step_names` | `list[str]` | All step names in execution order |
| `execution_levels` | `list[list[str]]` | Topological levels (steps within a level run in parallel) |

### Methods

| Method | Description |
|--------|-------------|
| `get_step(name)` | Get a step by name |
| `execute(rows, all_fields, config, ...)` | Execute pipeline across all rows (async) |

### Execution Model

```
Level 0: [step_a, step_b]  <- run in parallel (asyncio.gather)
Level 1: [step_c]          <- depends on step_a and/or step_b
Level 2: [step_d]          <- depends on step_c
```

Within each step, rows are processed concurrently (bounded by `config.max_workers`).

---

## LLMStep

Calls an OpenAI-compatible chat model to produce enrichment values. Uses `response_format={"type": "json_object"}` and validates responses with Pydantic.

```python
from lattice import LLMStep

step = LLMStep(
    name="analyze",
    fields={
        "market_size": "Estimate TAM in billions USD",
        "competition_level": {
            "prompt": "Rate competition level",
            "enum": ["Low", "Medium", "High"],
        },
    },
    max_retries=2,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Step name (unique within pipeline) |
| `fields` | `list[str] \| dict[str, str \| dict]` | required | Field names or inline 7-key field specs |
| `depends_on` | `list[str]` | `[]` | Step names this depends on |
| `model` | `str` | `"gpt-4.1-mini"` | OpenAI model name |
| `temperature` | `float` | `None` | Falls back to config, then `0.2` |
| `max_tokens` | `int` | `None` | Falls back to config, then `4000` |
| `system_prompt` | `str` | `None` | Overrides auto-generated prompt instructions (data sections still appended) |
| `api_key` | `str` | `None` | Falls back to `OPENAI_API_KEY` env var |
| `base_url` | `str` | `None` | OpenAI-compatible endpoint (Ollama, Groq, etc.) |
| `client` | `LLMClient` | `None` | Any `LLMClient` protocol adapter |
| `schema` | `Type[BaseModel]` | `EnrichmentResult` | Pydantic model for response validation |
| `max_retries` | `int` | `2` | Retries on JSON/validation errors (error fed back to LLM) |

### Field Specs (7-key)

When `fields` is a dict, each value is validated as a `FieldSpec` with these keys:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `prompt` | `str` | Yes | The extraction instruction |
| `type` | `str` | No | `String` (default), `Number`, `Boolean`, `Date`, `List[String]`, `JSON` |
| `format` | `str` | No | Output format pattern (e.g. `"YYYY-MM-DD"`, `"$X.XB"`) |
| `enum` | `list[str]` | No | Constrained value list |
| `examples` | `list[str]` | No | Good output examples |
| `bad_examples` | `list[str]` | No | Anti-patterns to avoid |
| `default` | `Any` | No | Fallback value when data is insufficient (enforced in Python) |

Unknown keys are rejected at construction time (`extra="forbid"`).

### Behavior

- Lazy client initialization (no import-time API key check)
- Dynamic system prompt: markdown headers + XML data boundaries (GPT-4.1 cookbook)
- Only describes field spec keys actually used across the step's fields
- Default enforcement: replaces LLM refusal text with field `default` values
- On parse/validation error: appends error to conversation and retries
- Returns `StepResult` with values filtered to declared fields only

---

## FunctionStep

Wraps any sync or async callable as a pipeline step. The escape hatch for APIs, databases, and custom logic.

```python
from lattice import FunctionStep

def lookup_funding(ctx):
    company = ctx.row.get("name")
    return {"funding_amount": call_api(company)}

step = FunctionStep(
    name="funding",
    fn=lookup_funding,
    fields=["funding_amount"],
    depends_on=["search"],  # optional
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Step name (unique within pipeline) |
| `fn` | `Callable` | `(StepContext) -> dict[str, Any]`. Sync functions run via `run_in_executor`. |
| `fields` | `list[str]` | Field names this step produces |
| `depends_on` | `list[str]` | Step names this depends on (default: `[]`) |

---

## Step Protocol

The interface all steps must satisfy. Uses `typing.Protocol` for duck typing — no inheritance required.

```python
from lattice import Step, StepContext, StepResult

@runtime_checkable
class Step(Protocol):
    name: str
    fields: list[str]
    depends_on: list[str]

    async def run(self, ctx: StepContext) -> StepResult: ...
```

### StepContext

Immutable context passed to each step's `run()` method.

| Field | Type | Description |
|-------|------|-------------|
| `row` | `dict[str, Any]` | Original row data |
| `fields` | `dict[str, dict[str, Any]]` | Field specs for this step only |
| `prior_results` | `dict[str, Any]` | Merged outputs from dependency steps |
| `config` | `Any` | EnrichmentConfig (optional) |

### StepResult

Output from a single step execution.

| Field | Type | Description |
|-------|------|-------------|
| `values` | `dict[str, Any]` | Field name -> produced value |
| `usage` | `UsageInfo` | Token usage (LLM steps only, optional) |
| `metadata` | `dict[str, Any]` | Arbitrary metadata |

---

## FieldManager

Loads field definitions from CSV. Provides category-based access to field specs.

```python
from lattice import FieldManager

fm = FieldManager.from_csv("field_categories.csv")

categories = fm.get_categories()          # ["business_analysis", "web_intelligence"]
fields = fm.get_category_fields("business_analysis")  # {"market_size": {...}, ...}
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `from_csv(path)` | `FieldManager` | Load field definitions from CSV |
| `get_categories()` | `list[str]` | List all available categories |
| `get_category_fields(category)` | `dict` | Get field specs for a category |
| `validate_category(category)` | `bool` | Check if category exists |
| `get_field_count(category=None)` | `int` | Count fields (per category or total) |

---

## EnrichmentConfig

Configuration dataclass with factory presets.

```python
from lattice import EnrichmentConfig

config = EnrichmentConfig(
    max_workers=30,             # Tier 2+ accounts
    temperature=0.2,            # Precise structured output
    enable_checkpointing=True,
)
```

See [Configuration Guide](configuration.md) for all options and presets.

---

## Error Handling

```python
from lattice import EnrichmentError, FieldValidationError
from lattice.core.exceptions import StepError, PipelineError

try:
    result = enricher.run(df, "business_analysis")
except FieldValidationError as e:
    print(f"Field routing error: {e}")
except StepError as e:
    print(f"Step '{e.step_name}' failed: {e}")
except PipelineError as e:
    print(f"Pipeline construction error: {e}")
except EnrichmentError as e:
    print(f"Enrichment error: {e}")
```

### Exception Hierarchy

| Exception | When Raised |
|-----------|-------------|
| `EnrichmentError` | Base exception for all Lattice errors |
| `FieldValidationError` | Invalid field definition, missing category, or field routing error |
| `StepError` | A step failed after exhausting retries |
| `PipelineError` | Duplicate names, missing dependencies, or cycles in pipeline |

---

## Schemas

Pydantic models used for LLM response validation.

### EnrichmentResult

Dynamic container for validated LLM responses. Accepts any fields.

```python
from lattice.schemas import EnrichmentResult

result = EnrichmentResult(market_size="$50B", competition_level="High")
print(result.model_dump())  # {"market_size": "$50B", "competition_level": "High"}
```

### FieldSpec

Strict Pydantic model for the 7-key field specification. Used internally by LLMStep; also available for direct validation.

```python
from lattice import FieldSpec

spec = FieldSpec(prompt="Estimate TAM", type="Number", format="$X.XB")
spec.model_dump(exclude_none=True)  # {"prompt": "Estimate TAM", "type": "Number", "format": "$X.XB"}
```
