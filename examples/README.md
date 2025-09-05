# Lattice Examples

This directory contains example configurations and sample data to help you get started with Lattice.

## Files

### Configuration Files (Always Committed)
- **`field_categories.csv`** - Example field definitions for different enrichment categories
- **`sample_data.csv`** - Sample company data for testing
- **`test_web_enrichment.py`** - Example script demonstrating web-enhanced enrichment

### Field Categories

#### `business_analysis`
Traditional business analysis fields using LLM reasoning:
- `market_size` - Total addressable market estimation
- `competition_level` - Competitive landscape assessment  
- `growth_potential` - Business growth evaluation

#### `web_intelligence` 
Web-enhanced fields using real-time search (requires Tavily API):
- `recent_funding` - Latest funding rounds and investment information
- `current_news` - Recent news articles and developments
- `executive_team` - Current leadership team and key executives
- `company_valuation` - Latest valuation or market cap information
- `market_trends` - Current market trends affecting the company

## Usage

### Basic Usage
```python
from lattice import TableEnricher, FieldManager, LLMChain

# Load field definitions
field_manager = FieldManager.from_csv("examples/field_categories.csv")

# Create LLM chain  
chain = LLMChain.openai(api_key="your-openai-key")

# Create enricher
enricher = TableEnricher(chain=chain, field_manager=field_manager)

# Load and enrich data
import pandas as pd
df = pd.read_csv("examples/sample_data.csv")
result = enricher.enrich_dataframe(df, category="business_analysis")
```

### Web-Enhanced Usage
```python
from lattice.chains import WebEnrichedLLMChain

# Create web-enhanced chain (requires Tavily API key)
web_chain = WebEnrichedLLMChain.create(
    api_key="your-openai-key",
    tavily_api_key="your-tavily-key"
)

# Use web intelligence fields
enricher = TableEnricher(chain=web_chain, field_manager=field_manager)
result = enricher.enrich_dataframe(df, category="web_intelligence")
```

### Running the Example
```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"  # Optional, for web features

# Run the demonstration
python examples/test_web_enrichment.py
```

## API Keys

### Required
- **OpenAI API Key**: Get from https://platform.openai.com/api-keys

### Optional (for web features)
- **Tavily API Key**: Get from https://tavily.com (free tier available)

## Output Files

Generated output files are saved to the `/data` directory and are git-ignored:
- `data/web_enriched_sample_data.csv` - Results from web enrichment demo
- `data/*` - All generated files

## Customization

### Adding Custom Fields
Edit `field_categories.csv` to add your own enrichment fields:

```csv
Category,Field,Prompt,Instructions,Data_Type,Example_1
my_category,my_field,Analyze this aspect of the company,Provide detailed analysis,String,Example output
```

### Custom Sample Data
Replace `sample_data.csv` with your own company data. Required columns:
- `name` - Company name
- `description` - Company description
- `industry` - Industry sector
- `location` - Company location

Additional columns will be passed as context to the LLM for analysis.