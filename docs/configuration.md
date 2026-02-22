# Configuration Guide

Complete reference for configuring Accrue enrichment pipelines.

## Quick Start

Use factory presets for common scenarios:

```python
from accrue import EnrichmentConfig

# Development: low concurrency, verbose logging (safe for Tier 1)
config = EnrichmentConfig.for_development()

# Production: high concurrency, checkpointing (Tier 2+)
config = EnrichmentConfig.for_production()

# Server: FastAPI/async context, no progress bars (Tier 2+)
config = EnrichmentConfig.for_server()
```

## All Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| **LLM** | | | |
| `max_tokens` | `int` | 4000 | Maximum tokens for LLM output |
| `temperature` | `float` | 0.2 | LLM temperature. Low (0.1-0.3) is best for structured enrichment |
| **Concurrency** | | | |
| `max_workers` | `int` | 10 | Concurrent rows per step (semaphore-bounded). Production uses 20-30. |
| **Fields** | | | |
| `overwrite_fields` | `bool` | False | Overwrite existing field values in DataFrame |
| **Reliability** | | | |
| `max_retries` | `int` | 3 | Maximum retry attempts for API errors (429, 500, timeouts) |
| `retry_base_delay` | `float` | 1.0 | Base delay for exponential backoff (seconds) |
| **Checkpointing** | | | |
| `enable_checkpointing` | `bool` | False | Save results after each step for crash recovery |
| `checkpoint_dir` | `str` | None | Directory for checkpoint files |
| `auto_resume` | `bool` | True | Automatically resume from checkpoint on re-run |
| **Caching (Phase 3)** | | | |
| `enable_caching` | `bool` | False | Enable input-hash cache to skip redundant API calls |
| `cache_ttl` | `int` | 3600 | Cache time-to-live in seconds |
| **Logging** | | | |
| `log_level` | `str` | "INFO" | Logging verbosity |
| `enable_progress_bar` | `bool` | True | Show tqdm progress bar during execution |

## Concurrency & Rate Limits

`max_workers` controls how many rows are processed concurrently within each step via `asyncio.Semaphore`. Set this based on your API provider's rate limits:

| Provider Tier | RPM | TPM | Recommended `max_workers` |
|---|---|---|---|
| OpenAI Tier 1 ($5 spend) | 500 | 200K | 5-10 |
| OpenAI Tier 2 ($50 spend) | 5,000 | 2M | 20-50 |
| OpenAI Tier 3+ ($100+) | 5,000+ | 4M+ | 50-100 |
| OpenAI Tier 5 ($1K+) | 30,000 | 150M | 100-200 |
| Anthropic Tier 1 ($5) | 50 | 30K | 5-10 |
| Anthropic Tier 2 ($40) | 1,000 | 450K | 20-50 |
| Anthropic Tier 4 ($400) | 4,000 | 2M | 100-200 |

**Default is 10** — safe for most Tier 1-2 accounts. Production deployments on Tier 2+ should use 20-30.

If you hit rate limits, Accrue's retry/backoff (Phase 2) will handle 429 errors automatically. Setting `max_workers` too high just means more retries; setting it too low means wasted throughput.

## Custom Configuration

```python
config = EnrichmentConfig(
    max_workers=30,          # Tier 2+ accounts
    temperature=0.2,         # Precise structured output
    max_retries=5,           # More resilient
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints",
)

result = pipeline.run(df, config=config)
```

## Sync & Async Usage

```python
from accrue import Pipeline, EnrichmentConfig

# Sync (wraps asyncio.run)
result = pipeline.run(df, config=config)

# Async (FastAPI, Jupyter, etc.)
result = await pipeline.run_async(df, config=config)

# Reusable runner for repeated execution
runner = pipeline.runner(config=config)
result = await runner.run_async(df)
```

## Checkpointing

Accrue checkpoints at the step level. After each step completes for all rows, results are saved to disk. If a later step fails, re-running resumes from the last completed step.

```python
config = EnrichmentConfig(
    enable_checkpointing=True,
    auto_resume=True,
    checkpoint_dir="./checkpoints",
)

result = pipeline.run(df, config=config)

# If interrupted after step 1 completes but step 2 fails:
# Re-running will skip step 1 and resume from step 2
result = pipeline.run(df, config=config)
```

Checkpoint files are automatically cleaned up after successful completion.

## Factory Presets

### `for_development()`
Safe for Tier 1 accounts and quick iteration.
- `max_workers=5`, `temperature=0.2`
- `enable_progress_bar=True`, `log_level="DEBUG"`

### `for_production()`
High throughput with crash recovery. For Tier 2+ accounts.
- `max_workers=30`, `temperature=0.2`
- `enable_checkpointing=True`, `max_retries=5`

### `for_server()`
Async server context (FastAPI, etc.). No console output.
- `max_workers=30`, `temperature=0.2`
- `enable_progress_bar=False`, `max_retries=5`, `log_level="WARNING"`
