# Lattice Architecture Reference

## Target User Experience

### Simple Usage (Primary)
```python
from lattice import TableEnricher, LLMChain, FieldManager

# Dead simple - 4 lines
enricher = TableEnricher(
    chain=LLMChain.openai(api_key="sk-..."),
    field_manager=FieldManager.from_csv("business_fields.csv")
)
result = enricher.enrich_dataframe(df, category="business_analysis")
```

### Advanced Usage (Vector Store)
```python
from lattice import VectorStoreLLMChain
from lattice.vector_store import VectorStore

# Pluggable but concrete
enricher = TableEnricher(
    chain=VectorStoreLLMChain(
        vector_store=VectorStore.from_directory("./docs"),
        llm_chain=LLMChain.openai(api_key="...")
    ),
    field_manager=FieldManager.from_csv("fields.csv")
)
```

### FastAPI Integration (Future)
```python
from fastapi import FastAPI
from lattice import TableEnricher

app = FastAPI()
enricher = TableEnricher(...)  # Initialize once

@app.post("/enrich")
async def enrich_data(data: UploadFile):
    df = pd.read_csv(data.file)
    result = await enricher.enrich_dataframe_async(df, "business_analysis")
    return result.to_dict()
```

## Core Components Breakdown

### 1. TableEnricher (Orchestrator)
**Responsibility**: Coordinate the enrichment process
**Size**: ~100 lines (down from 285)

```python
class TableEnricher:
    def __init__(self, chain, field_manager, config=None):
        self.processor = RowProcessor(chain, field_manager)
        self.config = config or EnrichmentConfig()
    
    def enrich_dataframe(self, df, category) -> pd.DataFrame:
        # Simple orchestration logic
    
    async def enrich_dataframe_async(self, df, category) -> pd.DataFrame:
        # Async version with concurrent processing
```

### 2. RowProcessor (Extracted Logic)
**Responsibility**: Process individual rows through chains
**Size**: ~80 lines

```python
class RowProcessor:
    def __init__(self, chain, field_manager):
        self.chain = chain
        self.field_manager = field_manager
    
    def process_row(self, row: pd.Series, fields: Dict) -> Dict[str, Any]:
        # Current _process_row logic from TableEnricher
    
    async def process_row_async(self, row: pd.Series, fields: Dict) -> Dict[str, Any]:
        # Async version for batch processing
```

### 3. Chain Classes (LangChain Wrappers)
**Responsibility**: Wrap LangChain functionality with clean API

```python
class LLMChain:
    """Simple LangChain wrapper"""
    @classmethod
    def openai(cls, model="gpt-3.5-turbo", **kwargs):
        # Factory method - easy instantiation
    
    def invoke(self, input_data: Dict) -> Dict:
        # Sync LangChain invocation
    
    async def ainvoke(self, input_data: Dict) -> Dict:
        # Async LangChain invocation

class VectorStoreLLMChain:
    """Your existing vector store + LLM logic"""
    def __init__(self, vector_store, llm_chain):
        # Composition not inheritance
```

### 4. FieldManager (Refined)
**Responsibility**: Field definitions and validation
**Changes**: Minimal - it's already well-designed

### 5. Configuration (Unified)
**Responsibility**: All configuration in one place

```python
@dataclass
class EnrichmentConfig:
    # Core processing
    max_tokens: int = 5000
    temperature: float = 0.5
    row_delay: float = 1.0
    
    # Performance  
    batch_size: int = 10
    max_workers: int = 3
    
    # Features
    enable_caching: bool = False
    progress_callback: Optional[Callable] = None
```

## Data Flow

### Simple Enrichment Flow
```
DataFrame → TableEnricher → RowProcessor → LLMChain → LangChain → OpenAI → Results
    ↑              ↓              ↓           ↓           ↓        ↓        ↓
   Input       Orchestrate    Process     Format      API     Parse    Output
```

### Vector Store Enhanced Flow  
```
DataFrame → TableEnricher → RowProcessor → VectorStoreLLMChain → Results
                                ↓              ↓
                          VectorStore → LLMChain → LangChain → OpenAI
                               ↓           ↓         ↓         ↓
                           Search     Combine    Format      API
```

## Scalability Considerations

### Current Bottlenecks
- Sequential row processing (1-3 seconds per row)
- No caching for duplicate content
- Memory loading entire DataFrames

### Phase 2 Improvements
- Concurrent processing with semaphore control
- In-memory caching for duplicate rows
- Async/await throughout

### Phase 3 Scaling
- Redis-backed caching
- Streaming data processing 
- Rate limiting and retry logic
- Background job processing

## Extension Points

### 1. New Chain Types
```python
class WebSearchChain:
    """Real-time web search enrichment"""
    
class DatabaseChain:
    """Historical data enrichment"""

class APIChain:
    """External API data enrichment"""
```

### 2. Context Builders
```python
class ContextBuilder:
    def build_context(self, row_data: Dict) -> str:
        # Build context from various sources
```

### 3. Validators
```python
class OutputValidator:
    def validate(self, result: Dict) -> bool:
        # Validate enrichment results
```

### 4. Cache Backends
```python
class CacheBackend:
    def get(self, key: str) -> Optional[Dict]:
        # Retrieve cached results
    
    def set(self, key: str, value: Dict, ttl: int):
        # Store results
```

## Production Deployment Considerations

### FastAPI Service Architecture
```python
# Service-level concerns
class EnrichmentService:
    def __init__(self):
        self.enricher_pool = AsyncPool(max_workers=5)
        self.rate_limiter = RateLimiter(requests_per_minute=100)
        self.cache = RedisCache("redis://...")
    
    async def enrich(self, df: pd.DataFrame) -> EnrichmentResult:
        # Production-ready enrichment with error handling
```

### Key Requirements
1. **Async processing** - Non-blocking FastAPI endpoints
2. **Resource pooling** - Handle concurrent requests
3. **Progress tracking** - Long-running job status
4. **Error resilience** - Partial results, retry logic
5. **Caching** - Avoid duplicate API calls
6. **Rate limiting** - Respect API quotas
7. **Monitoring** - Logging, metrics, tracing

## Migration Path

### Phase 1: Clean Architecture
- Break up god classes
- Fix imports and package structure
- Add type hints
- Maintain current functionality

### Phase 2: Enhanced Features  
- Add async support
- Implement error handling
- Basic caching layer

### Phase 3: Production Ready
- FastAPI integration examples
- Advanced caching (Redis)
- Streaming support
- Comprehensive monitoring

## Success Metrics
- **API Simplicity**: `from lattice import TableEnricher` gets you started
- **Performance**: 10x faster processing with concurrent execution
- **Reliability**: Graceful handling of API failures with partial results
- **Extensibility**: Easy to add new chain types and context sources
- **Production Ready**: Drop-in library for FastAPI services