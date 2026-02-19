# Lattice Examples

This directory contains example configurations and sample data for Lattice.

## Files

- **`field_categories.csv`** - Example field definitions for enrichment categories
- **`sample_data.csv`** - Sample company data for testing

## Field Categories

### `business_analysis`
LLM-powered business analysis fields:
- `market_size` - Total addressable market estimation
- `competition_level` - Competitive landscape assessment
- `growth_potential` - Business growth evaluation

## Usage

### Single LLM Step
```python
import pandas as pd
from lattice import Enricher, Pipeline, LLMStep, FieldManager

pipeline = Pipeline([
    LLMStep("analyze", fields=["market_size", "competition_level", "growth_potential"])
])

field_manager = FieldManager.from_csv("examples/field_categories.csv")
enricher = Enricher(pipeline=pipeline, field_manager=field_manager)

df = pd.read_csv("examples/sample_data.csv")
result = enricher.run(df, category="business_analysis")
print(result)
```

### Multi-Step Pipeline
```python
from lattice import Enricher, Pipeline, FunctionStep, LLMStep, FieldManager

def lookup_funding(ctx):
    company = ctx.row.get("name", "")
    return {"funding_amount": f"Looked up {company}"}

pipeline = Pipeline([
    FunctionStep("lookup", fn=lookup_funding, fields=["funding_amount"]),
    LLMStep("analyze", fields=["market_size"], depends_on=["lookup"]),
])

field_manager = FieldManager.from_csv("examples/field_categories.csv")
enricher = Enricher(pipeline=pipeline, field_manager=field_manager)

df = pd.read_csv("examples/sample_data.csv")
result = enricher.run(df, category="business_analysis")
```

## API Keys

### Required
- **OpenAI API Key**: Set `OPENAI_API_KEY` in your environment or `.env` file

## Customization

### Adding Custom Fields
Edit `field_categories.csv` to add your own enrichment fields:

```csv
Category,Field,Prompt,Instructions,Data_Type,Example_1
my_category,my_field,Analyze this aspect,Provide detailed analysis,String,Example output
```

### Custom Sample Data
Replace `sample_data.csv` with your own data. Any columns present will be passed as row context to pipeline steps.
