# Lattice - Enrichment Pipeline Engine

A composable pipeline engine for structured data enrichment using LLMs. Define a pipeline of steps, point it at a DataFrame, and get structured results. Lattice handles orchestration: column-oriented batching, step dependencies, Pydantic validation, retries, checkpointing, and async concurrency.

## Features

- **Composable pipeline** - Chain LLM calls, API lookups, and custom functions as steps with declared dependencies
- **Column-oriented execution** - Each step runs across ALL rows before the next step starts; independent steps run in parallel
- **Step protocol** - Simple async `run()` interface; bring your own step classes via duck typing
- **Pydantic validation** - LLM responses validated against schemas with automatic retry on parse errors
- **Per-step checkpointing** - Resume interrupted pipelines from the last completed step
- **Field routing validation** - Fail fast if pipeline steps don't cover the requested category fields
- **Async concurrency** - Configurable worker count with semaphore-bounded parallelism

## Prerequisites

- Python 3.9+
- OpenAI API key

## Installation

```bash
pip install -e .
```

Set up environment variables:
```bash
echo 'OPENAI_API_KEY=your_key_here' > .env
```

## Quick Start

```python
import pandas as pd
from lattice import Enricher, Pipeline, LLMStep, FieldManager

# Define a pipeline with one LLM step
pipeline = Pipeline([
    LLMStep("analyze", fields=["market_size", "competition_level", "growth_potential"])
])

# Load field definitions and create enricher
field_manager = FieldManager.from_csv("examples/field_categories.csv")
enricher = Enricher(pipeline=pipeline, field_manager=field_manager)

# Enrich a DataFrame
df = pd.read_csv("examples/sample_data.csv")
result = enricher.run(df, category="business_analysis")
```

## Multi-Step Pipeline

```python
from lattice import Pipeline, FunctionStep, LLMStep

def lookup_funding(ctx):
    company = ctx.row.get("name")
    return {"funding_amount": call_crunchbase_api(company)}

pipeline = Pipeline([
    FunctionStep("crunchbase", fn=lookup_funding, fields=["funding_amount"]),
    LLMStep("analysis", fields=["investment_thesis"], depends_on=["crunchbase"]),
])
```

Steps declare dependencies. Independent steps run in parallel; dependent steps wait for their inputs.

## Architecture

```
[All rows] -> Step 1 (function)  -> batch complete
[All rows] -> Step 2 (LLM call)  -> batch complete
[All rows] -> Step 3 (LLM call)  -> batch complete
```

Each step produces named fields. Fields prefixed with `__` are internal (inter-step communication) and filtered from the output DataFrame.

### Package Structure
```
lattice/
├── steps/       # Step protocol + built-in steps (LLMStep, FunctionStep)
├── pipeline/    # DAG resolution + column-oriented execution
├── schemas/     # Pydantic models (EnrichmentSpec, EnrichmentResult, StructuredResult)
├── core/        # Enricher, config, checkpoint, exceptions
├── data/        # FieldManager (CSV field definitions)
└── utils/       # Logging
```

## Configuration

```python
from lattice import EnrichmentConfig

config = EnrichmentConfig(
    max_workers=5,              # Concurrent rows per step
    enable_checkpointing=True,  # Save after each step completes
    overwrite_fields=False,     # Preserve existing field values
)

enricher = Enricher(pipeline=pipeline, field_manager=fm, config=config)
```

See [Configuration Guide](docs/configuration.md) for all options.

## Field Categories CSV

Define enrichment fields in a CSV file:

```csv
Category,Field,Prompt,Instructions,Data_Type,Example_1
business_analysis,market_size,Estimate the total addressable market,Provide in billions USD,String,$50B - Cloud infrastructure
business_analysis,competition_level,Assess competition level,Rate as Low/Medium/High,String,High - Competes with AWS
```

## Documentation

- [API Reference](docs/api-reference.md) - Detailed component documentation
- [Configuration Guide](docs/configuration.md) - All configuration options
- [Pipeline Design](docs/instructions/PIPELINE_DESIGN.md) - Architecture specification
- [GitHub Standards](docs/github-standards.md) - Contribution workflow

## License

MIT
