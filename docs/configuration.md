# Configuration Guide

Complete reference for configuring Lattice enrichment pipelines.

## Quick Start

Use factory presets for common scenarios:

```python
from lattice import EnrichmentConfig

# Development: Fast iteration, verbose logging
config = EnrichmentConfig.for_development()

# Production: Rate limiting, retries, checkpointing
config = EnrichmentConfig.for_production()

# FastAPI: Async-first, no progress bars
config = EnrichmentConfig.for_fast_api()
```

## All Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| **LLM** | | | |
| `max_tokens` | `int` | 4000 | Maximum tokens for LLM output |
| `temperature` | `float` | 0.5 | LLM temperature (0.0-2.0) |
| `context_window` | `int` | 20000 | Context window size |
| **Concurrency** | | | |
| `max_workers` | `int` | 3 | Concurrent rows per step (semaphore-bounded) |
| `row_delay` | `float` | 1.0 | Delay between rows in seconds |
| `batch_size` | `int` | 10 | Rows per batch |
| **Fields** | | | |
| `overwrite_fields` | `bool` | False | Overwrite existing field values in DataFrame |
| **Reliability** | | | |
| `enable_retries` | `bool` | True | Auto-retry on failures |
| `max_retries` | `int` | 3 | Maximum retry attempts |
| `retry_delay` | `float` | 2.0 | Delay between retries (seconds) |
| **Checkpointing** | | | |
| `enable_checkpointing` | `bool` | False | Save progress after each step completes |
| `checkpoint_dir` | `str` | None | Directory for checkpoint files |
| `auto_resume` | `bool` | True | Automatically resume from checkpoint |
| `checkpoint_interval` | `int` | 100 | Checkpoint save interval |
| **Logging** | | | |
| `log_level` | `str` | "INFO" | Logging verbosity |
| `log_dir` | `str` | None | Directory for log files |
| `enable_progress_bar` | `bool` | True | Show progress bars |

## Custom Configuration

```python
config = EnrichmentConfig(
    # LLM settings
    max_tokens=4000,
    temperature=0.3,

    # Concurrency
    max_workers=5,

    # Fields
    overwrite_fields=True,

    # Checkpointing
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints",
    auto_resume=True,

    # Logging
    log_level="DEBUG",
)
```

## Async Usage

The `Enricher` provides both sync and async entry points:

```python
from lattice import Enricher, EnrichmentConfig

config = EnrichmentConfig(max_workers=5)
enricher = Enricher(pipeline=pipeline, field_manager=fm, config=config)

# Sync (wraps asyncio.run — cannot be called inside async context)
result = enricher.run(df, "business_analysis")

# Async (use directly in async contexts like FastAPI)
result = await enricher.run_async(df, "business_analysis")
```

`max_workers` controls how many rows are processed concurrently within each step via an `asyncio.Semaphore`.

## Checkpointing

Lattice checkpoints at the step level, not the row level. After each step completes for all rows, its results are saved to disk. If a later step fails, re-running resumes from the last completed step.

```python
config = EnrichmentConfig(
    enable_checkpointing=True,
    auto_resume=True,
    checkpoint_dir="./checkpoints",
)

enricher = Enricher(pipeline=pipeline, field_manager=fm, config=config)
result = enricher.run(df, "business_analysis")

# If interrupted after step 1 completes but step 2 fails:
# Re-running will skip step 1 and resume from step 2
result = enricher.run(df, "business_analysis")
```

Checkpoint files are automatically cleaned up after successful completion.

## Factory Presets

### `for_development()`
- `row_delay=0.5`, `batch_size=5`, `max_workers=2`
- `enable_progress_bar=True`, `log_level="DEBUG"`

### `for_production()`
- `row_delay=2.0`, `batch_size=20`, `max_workers=5`
- `enable_async=True`, `enable_caching=True`, `enable_retries=True`

### `for_fast_api()`
- `enable_async=True`, `batch_size=50`, `max_workers=10`
- `enable_progress_bar=False`, `log_level="WARNING"`
