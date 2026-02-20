# Lattice Examples

This directory contains example configurations and sample data for Lattice.

## Files

- **`demo.ipynb`** - Interactive notebook demonstrating all Phase 3 features
- **`field_categories.csv`** - Example field definitions for CSV-based field loading
- **`sample_data.csv`** - Sample company data for testing

## Quick Start

### Inline field specs (recommended)
```python
import pandas as pd
from lattice import Pipeline, LLMStep

pipeline = Pipeline([
    LLMStep("analyze", fields={
        "market_size": "Estimate TAM in billions USD",
        "competition": {
            "prompt": "Rate competition level with key competitors",
            "enum": ["Low", "Medium", "High"],
            "default": "Unknown",
        },
    })
])

df = pd.read_csv("examples/sample_data.csv")
result = pipeline.run(df)
print(result.data)
```

### Multi-step pipeline
```python
from lattice import Pipeline, FunctionStep, LLMStep

def lookup_funding(ctx):
    company = ctx.row.get("name", "")
    return {"funding_amount": f"Looked up {company}"}

pipeline = Pipeline([
    FunctionStep("lookup", fn=lookup_funding, fields=["funding_amount"]),
    LLMStep("analyze", fields={
        "investment_thesis": "Write a one-sentence investment thesis using the funding data",
    }, depends_on=["lookup"]),
])

result = pipeline.run(df)
```

### CSV field loading (for teams)
```python
from lattice import Pipeline, LLMStep
from lattice.data import load_fields

fields = load_fields("examples/field_categories.csv", category="business_analysis")
pipeline = Pipeline([LLMStep("analyze", fields=fields)])
result = pipeline.run(df)
```

## API Keys

### Required
- **OpenAI API Key**: Set `OPENAI_API_KEY` in your environment or `.env` file

## CSV Field Format

```csv
Category,Field,Prompt,Type,Enum,Examples
my_category,my_field,Analyze this aspect,String,,Example output
my_category,risk,Rate risk level,String,"Low, Medium, High",High
```

Required columns: `Category`, `Field`, `Prompt`. Optional: `Type`, `Format`, `Enum`, `Examples`, `Bad_Examples`, `Default`.

Legacy columns (`Instructions`, `Data_Type`) are supported for backward compatibility.
