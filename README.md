# Lattice - CSV Enrichment Tool

A powerful tool for enriching CSV data using Large Language Models (LLM) on a row-by-row basis. This tool allows you to automatically analyze and enhance your tabular data with AI-generated insights.

## Features

- **Row-by-row LLM processing** - Processes each CSV row individually with configurable delays
- **Field-based enrichment** - Define custom fields with specific prompts and data types
- **Category-based organization** - Group related fields into categories for organized processing
- **Web-enhanced intelligence** - Real-time web search integration via Tavily API
- **Modern LLM support** - Uses latest OpenAI models (gpt-4o, gpt-4o-mini) with 8K tokens
- **Progress tracking** - Real-time progress bars and detailed logging
- **Configurable LLM settings** - Customizable temperature, token limits, and delays
- **Error handling** - Robust error handling with graceful fallbacks
- **Incremental processing** - Skip already processed fields to resume interrupted runs

## Current State

The tool is in a working state with basic LLM enrichment functionality. Vector store integration is present but not currently used in the test setup.

## Prerequisites

- Python 3.9+
- OpenAI API key (required)
- Tavily API key (optional, for web search features)
- Virtual environment (recommended)

## Installation

1. **Clone/navigate to the project directory:**
```bash
cd /path/to/lattice
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install pandas langchain-core langchain-openai langchain tqdm tenacity python-dotenv colorlog python-json-logger PyMuPDF ipython
```

4. **Set up environment variables:**
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Quick Start

1. **Run the test script:**
```bash
python test_enrichment.py
```

This will:
- Load sample company data (5 companies)
- Process them through the "business_analysis" category
- Generate insights for market size, competition level, and growth potential
- Save results to `enriched_sample_data.csv`

## Configuration Files

### Field Categories (`field_categories.csv`)
Defines the fields to enrich with their prompts and specifications:

```csv
Category,Field,Prompt,Instructions,Data_Type,Example_1
business_analysis,market_size,Estimate the total addressable market size for this company,Provide market size in billions USD and brief reasoning,String,"$50B - Cloud infrastructure market is rapidly growing"
business_analysis,competition_level,Assess the level of competition in this industry,Rate as Low/Medium/High and provide 1-2 key competitors,String,"High - Competes with AWS Amazon Web Services"
business_analysis,growth_potential,Evaluate the growth potential for this business,Rate as Low/Medium/High and provide reasoning,String,"High - Cloud adoption accelerating post-pandemic"
```

### Sample Data (`sample_data.csv`)
Input CSV with company data:

```csv
uuid,name,description,industry,location
1,Acme Corp,Technology company specializing in cloud solutions,Technology,San Francisco
2,Global Foods,International food distribution company,Food & Beverage,New York
```

## Key Components

### TableEnricher
Main class that handles the enrichment process:
- Processes categories of fields
- Manages row-by-row iteration
- Handles progress saving and error recovery

### FieldManager
Manages field categories and specifications:
- Loads field definitions from CSV
- Provides category-based field access
- Handles field examples and validation

### SimpleLLMChain
Basic LLM chain for enrichment:
- Interfaces with OpenAI GPT models
- Handles prompt formatting and response parsing
- Provides error handling for LLM calls

## Configuration Options

Edit `config.py` to adjust:
- `max_tokens`: Maximum tokens for LLM output (default: 5000)
- `temperature`: LLM temperature 0.0-1.0 (default: 0.5)
- `row_delay`: Delay between processing rows in seconds (default: 2.0)
- `context_window`: Size of context window for LLM (default: 20000)

## Quick Start

See the `examples/` directory for complete working examples:

- **`examples/field_categories.csv`** - Sample field definitions
- **`examples/sample_data.csv`** - Test company data  
- **`examples/test_web_enrichment.py`** - Full demonstration script
- **`examples/README.md`** - Detailed usage instructions

Run the example:
```bash
python examples/test_web_enrichment.py
```

## Usage Examples

### Basic Enrichment
```python
from enrichment import TableEnricher
from field_manager import FieldManager
from simple_chain import SimpleLLMChain

# Initialize components
field_manager = FieldManager("field_categories.csv")
chain = SimpleLLMChain()
enricher = TableEnricher(chain, field_manager=field_manager)

# Process data
df = pd.read_csv("your_data.csv")
enriched_df = enricher.process_category(df, "business_analysis")
```

### Custom Field Categories
Add new categories to `field_categories.csv`:
```csv
Category,Field,Prompt,Instructions,Data_Type
financial_analysis,revenue_estimate,Estimate annual revenue,Provide estimate in millions USD,String
financial_analysis,profit_margin,Estimate profit margin,Provide percentage and reasoning,String
```

## Output

The enriched data includes original columns plus new fields with LLM-generated insights:

```csv
uuid,name,industry,market_size,competition_level,growth_potential
1,Acme Corp,Technology,"$100B - Cloud solutions market expanding","High - Competes with AWS, Azure","High - Strong growth trajectory"
```

## Logging

The tool provides comprehensive logging:
- Row-by-row processing status
- API call information
- Error handling and recovery
- Progress tracking with completion times

## Limitations

- Currently requires OpenAI API (GPT-3.5-turbo or GPT-4)
- Vector store functionality present but not integrated in current test setup
- Response parsing assumes JSON format from LLM
- Sequential processing (one row at a time)

## Future Enhancements

- Vector store integration for context-aware enrichment
- Support for multiple LLM providers
- Batch processing capabilities
- Advanced response validation and parsing
- Web interface for configuration and monitoring

## Troubleshooting

**Import errors**: Make sure all dependencies are installed in your virtual environment

**API errors**: Verify your OpenAI API key is set correctly in `.env`

**Memory issues**: Adjust `max_tokens` and `context_window` in config for large datasets

**Rate limits**: Increase `row_delay` to avoid hitting API rate limits

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]