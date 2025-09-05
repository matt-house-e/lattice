# Lattice Refactoring Implementation Plan

## Overview
Transform the existing CSV enrichment tool into a clean, production-ready library with clear separation of concerns, proper package structure, and FastAPI integration capabilities.

## Core Principles
- **No over-engineering**: Keep abstractions minimal and practical
- **Use LangChain**: Leverage existing LangChain integration, don't fight it
- **Composition over inheritance**: Break up god classes with focused components
- **Production-ready**: Build with async support and FastAPI integration in mind
- **Maintain functionality**: Ensure `test_enrichment.py` continues working

## Phase 1: Foundation & Core Architecture (Priority 1)

### 1.1 Fix Package Structure
```
lattice/
├── __init__.py              # Clean public API
├── enricher.py             # TableEnricher (orchestrator)  
├── processors.py           # RowProcessor (extracted from TableEnricher)
├── chains.py              # LLMChain, VectorStoreLLMChain (simplified)
├── fields.py              # FieldManager (renamed, cleaned up)
├── config.py              # Single unified config class
├── exceptions.py          # Custom exceptions
├── utils.py               # Only keep what's actually used
├── vector_store/          # Keep existing, make importable
└── tests/
    └── test_enrichment.py  # Updated to use new API
```

### 1.2 Core Classes Design

#### TableEnricher (Main Orchestrator)
```python
class TableEnricher:
    """Main orchestrator - much smaller now, focuses on coordination"""
    def __init__(self, chain, field_manager, config=None):
        self.processor = RowProcessor(chain, field_manager)  
        self.config = config or EnrichmentConfig()
    
    def enrich_dataframe(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Sync version for simple use cases"""
    
    async def enrich_dataframe_async(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Async version for FastAPI integration"""
```

#### RowProcessor (Extracted Logic)
```python
class RowProcessor:
    """Handles single row processing logic (extracted from TableEnricher)"""
    def __init__(self, chain, field_manager):
        self.chain = chain
        self.field_manager = field_manager
    
    def process_row(self, row: pd.Series, fields: Dict) -> Dict[str, Any]:
        """Process one row through the chain"""
    
    async def process_row_async(self, row: pd.Series, fields: Dict) -> Dict[str, Any]:
        """Async version for batch processing"""
```

#### Chain Classes (LangChain Wrappers)
```python
class LLMChain:
    """Simple LangChain wrapper with factory methods"""
    def __init__(self, llm: BaseLanguageModel, prompt_template: ChatPromptTemplate):
        self.llm = llm
        self.prompt_template = prompt_template
    
    @classmethod
    def openai(cls, model="gpt-3.5-turbo", api_key=None, **kwargs):
        """Factory method for OpenAI models"""
    
    def invoke(self, input_data: Dict) -> Dict:
        """Synchronous invocation"""
    
    async def ainvoke(self, input_data: Dict) -> Dict:
        """Async invocation for FastAPI"""

class VectorStoreLLMChain:
    """Combines vector store + LLM (existing logic, cleaned up)"""
    def __init__(self, vector_store, llm_chain):
        self.vector_store = vector_store
        self.llm_chain = llm_chain
```

### 1.3 Configuration Unification
```python
@dataclass
class EnrichmentConfig:
    """Single config class replacing scattered configurations"""
    # Processing
    max_tokens: int = 5000
    temperature: float = 0.5
    row_delay: float = 1.0
    batch_size: int = 10
    max_workers: int = 3
    
    # Async/FastAPI
    enable_async: bool = False
    progress_callback: Optional[Callable] = None
    
    # Caching (for future)
    cache_enabled: bool = False
    cache_ttl: int = 3600
```

### 1.4 Clean Public API
```python
# lattice/__init__.py
from .enricher import TableEnricher
from .chains import LLMChain, VectorStoreLLMChain  
from .fields import FieldManager
from .config import EnrichmentConfig
from .exceptions import EnrichmentError, FieldValidationError

__version__ = "0.2.0"
__all__ = [
    'TableEnricher', 
    'LLMChain', 
    'VectorStoreLLMChain',
    'FieldManager', 
    'EnrichmentConfig'
]
```

## Phase 2: Enhanced Functionality (Priority 2)

### 2.1 Async Support for FastAPI
- Add async methods to all core classes
- Implement concurrent row processing with semaphore control
- Add progress tracking via callbacks

### 2.2 Error Handling & Resilience
```python
class EnrichmentError(Exception):
    """Base exception for enrichment errors"""
    def __init__(self, message: str, row_index: Optional[int] = None):
        self.message = message
        self.row_index = row_index
        super().__init__(message)

class PartialEnrichmentResult:
    """Container for results with some failures"""
    def __init__(self, df: pd.DataFrame, errors: List[EnrichmentError]):
        self.df = df
        self.errors = errors
        self.success_rate = len(df) / (len(df) + len(errors))
```

### 2.3 Basic Caching Layer
- Simple in-memory cache for duplicate row content
- Redis integration for FastAPI deployments

## Phase 3: Production Features (Priority 3)

### 3.1 Streaming Support
```python
async def enrich_stream(self, data_stream) -> AsyncGenerator[pd.DataFrame]:
    """Process large datasets without loading all into memory"""
```

### 3.2 Advanced Context Management
- Pluggable context builders
- Smart vector store integration
- External API context sources

### 3.3 Validation & Quality Control
- Built-in validators for common data types
- LLM-based validation for output quality
- Confidence scoring

## Implementation Order

### Week 1: Foundation
1. **Fix imports and package structure** - Remove all try/except import blocks
2. **Extract RowProcessor** from TableEnricher 
3. **Unify configuration** - Single EnrichmentConfig class
4. **Clean up chains** - Simplify LangChain wrappers
5. **Update test_enrichment.py** - Ensure it still passes

### Week 2: Core Features  
6. **Add proper type hints** throughout codebase
7. **Implement custom exceptions** with helpful error messages
8. **Add async support** to core classes
9. **Basic progress tracking** via callbacks

### Week 3: Production Ready
10. **Add caching layer** (in-memory first)
11. **Concurrent processing** with proper rate limiting
12. **FastAPI integration examples**
13. **Unit tests** for core components

## Success Criteria
- [ ] `test_enrichment.py` passes with new architecture
- [ ] Clean, discoverable API: `from lattice import TableEnricher`
- [ ] Async support ready for FastAPI integration
- [ ] Proper error handling with partial results
- [ ] Type hints on all public methods
- [ ] Basic unit test coverage
- [ ] Documentation updated

## Reference Materials
- **Current codebase analysis**: See `CODE_REVIEW_REPORT.md`
- **LangChain patterns**: Research as needed during implementation
- **FastAPI integration**: Build async-first where it makes sense

## Notes
- Keep vector store in-repo but make it pluggable
- Use LangChain but wrap it in simple, focused classes
- Don't create abstractions until you need them
- Focus on composition over complex inheritance hierarchies