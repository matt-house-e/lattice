# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-02-19

### Added
- **Pipeline architecture**: Composable DAG of steps with topological execution
- **Step protocol**: Async-only `run()` interface via `typing.Protocol` (duck typing, no inheritance required)
- **LLMStep**: Direct OpenAI SDK calls with JSON response format, Pydantic validation, and self-correcting retries
- **FunctionStep**: Wraps any sync or async callable as a pipeline step (sync functions run via `run_in_executor`)
- **Column-oriented execution**: Each step runs across ALL rows before the next step starts; independent steps run in parallel via `asyncio.gather`
- **Per-step checkpointing**: Save after each step completes; resume from last completed step on failure
- **Field routing validation**: Fail fast if pipeline steps don't cover all category fields or produce duplicates
- **Internal fields**: `__` prefix for inter-step communication, filtered from output DataFrame
- **Overwrite control**: `overwrite_fields` parameter preserves existing values when `False`
- 80 unit and integration tests covering pipeline, steps, enricher, and checkpointing

### Changed
- **BREAKING**: Replaced `TableEnricher` with `Enricher(pipeline, field_manager, config)`
- **BREAKING**: Replaced `LLMChain` / `WebEnrichedLLMChain` with `LLMStep` / `FunctionStep` in a `Pipeline`
- **BREAKING**: Processing model changed from row-oriented to column-oriented
- **BREAKING**: Checkpoint format changed from per-row index to per-step results
- `EnrichmentConfig` now uses `max_workers` for concurrency control (semaphore-bounded)
- System prompt ported from v0.2 `LLMChain` to `LLMStep` default prompt (~50 lines of enrichment-specific prompt engineering)

### Removed
- `lattice/chains/` module (LLMChain, WebEnrichedLLMChain) — replaced by `lattice/steps/`
- `lattice/vector_store/` module (FAISS, document processing) — dead code
- `lattice/core/processors.py` (RowProcessor) — replaced by `lattice/pipeline/`
- LangChain dependencies (`langchain`, `langchain-core`, `langchain-openai`)
- `tenacity` dependency (retry logic now built into LLMStep)
- `tavily-python` dependency (web search deferred to Phase 2)
- Vector store optional dependencies (`faiss-cpu`, `pymupdf`)
- CLI entry point (`lattice-enrich`) — no CLI module exists yet

### Fixed
- `fields.py` examples column bug: position-based `df.columns[5:]` slicing replaced with name-based `KNOWN_COLUMNS` exclusion to prevent V2 CSV columns leaking into examples
- Added `pydantic>=2.0.0` as explicit dependency
- Removed dead `vector_store_path` and `similarity_threshold` from `EnrichmentConfig`

---

### Migration from v0.2

```python
# v0.2
from lattice import TableEnricher, LLMChain, FieldManager
chain = LLMChain.openai(api_key="sk-...")
enricher = TableEnricher(chain=chain, field_manager=fm)
result = enricher.enrich_dataframe(df, category="business_analysis")

# v0.3
from lattice import Enricher, Pipeline, LLMStep, FieldManager
pipeline = Pipeline([LLMStep("analyze", fields=["market_size", "competition_level"])])
enricher = Enricher(pipeline=pipeline, field_manager=fm)
result = enricher.run(df, category="business_analysis")
```

## [0.2.0] - 2024-09-05

### Added
- Complete architecture refactoring for production readiness
- Clean package structure with proper module organization
- Enhanced configuration system with development/production presets
- Comprehensive error handling with custom exceptions
- Async processing support for FastAPI integration
- Professional packaging with pyproject.toml
- Factory methods for easy chain creation

### Changed
- **BREAKING**: Refactored from flat file structure to proper package
- **BREAKING**: Simplified API - `TableEnricher` now uses composition pattern
- **BREAKING**: Unified configuration in `EnrichmentConfig` class

### Removed
- Redundant old implementation files
- Over-engineered logging system
- Unused utility functions and dead code

## [0.1.0] - 2024-09-05

### Added
- Initial working implementation
- Basic CSV enrichment with LangChain LLM processing
- Field category management system
- Vector store support (experimental)
- Progress tracking with tqdm
- Example data and test scripts
