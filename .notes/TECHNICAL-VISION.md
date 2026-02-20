# Lattice Technical Vision & Architecture

**Last Updated:** 2026-02-20

---

## TL;DR

Lattice evolves from simple row-by-row enrichment to a **knowledge synthesis engine** through five dimensions:
1. **Provenance & Trust** - Every value has sources and confidence
2. **Cross-Row Intelligence** - Dataset-aware enrichment
3. **Knowledge Graph** - Extract entity relationships
4. **Intelligence Fusion** - Multi-source orchestration
5. **Temporal Intelligence** - Change tracking and freshness

The engine provides **primitives**. Products (Lead Scorer, Shortlist) compose them into domain-specific intelligence.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRODUCTS                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Shortlist  │  │ Lead Scorer │  │   Competitive Intel     │ │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
│         └────────────────┼───────────────────────┘              │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                  LATTICE ENGINE (OSS)                      │ │
│  │                                                            │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │ │
│  │  │ Enricher │ │ Sources  │ │ Output   │ │   Hooks &    │  │ │
│  │  │  Core    │ │  Layer   │ │ Schema   │ │   Events     │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │ │
│  │  │  Cache   │ │  Costs   │ │ Streaming│ │ Cross-Row    │  │ │
│  │  │  Layer   │ │ Tracking │ │  Output  │ │ Intelligence │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Five Dimensions of Intelligence

### Current State: One-Dimensional

```
Input DataFrame → [LLM processes each row independently] → Output DataFrame
```

Row-centric, single-source, point-in-time, opaque, flat relationships.

### The Five Dimensions

| Dimension | What It Adds | Effort | Impact |
|-----------|--------------|--------|--------|
| **Provenance** | Sources, confidence, reasoning | Medium | High |
| **Cross-Row** | Dataset context, comparisons | Medium | High |
| **Knowledge Graph** | Entity relationships | High | Very High |
| **Intelligence Fusion** | Multi-source orchestration | High | Very High |
| **Temporal** | Change tracking, freshness | Medium | Medium-High |

---

## Dimension 1: Provenance & Trust

**Every enriched value includes metadata about where it came from.**

```python
# Current output
{"market_size": "$5B"}

# With provenance
{
    "market_size": {
        "value": "$5B",
        "confidence": 0.87,
        "sources": [
            {"type": "web", "url": "https://techcrunch.com/...", "title": "..."},
            {"type": "llm_knowledge", "model": "gpt-4o"}
        ],
        "reasoning": "TechCrunch cited Gartner report...",
        "cost_usd": 0.003
    }
}
```

**API:**
```python
enricher = TableEnricher(output_mode="rich")  # value + metadata columns
result = enricher.enrich(df, category="research")

# DataFrame includes: funding, funding_sources, funding_confidence
```

---

## Dimension 2: Cross-Row Intelligence

**LLM sees dataset context, enabling comparative analysis.**

### Approach A: Context Injection
```python
# LLM sees: "This dataset contains 50 B2B SaaS companies, $2M-$200M funding..."
enricher = TableEnricher(inject_context=True)
```

### Approach B: Two-Pass (Comparative)
```python
# Pass 1: Standard enrichment
# Pass 2: Add ranking, clustering, outliers

result = enricher.enrich_with_comparison(df, category="research")
# Adds: funding_rank, cluster, most_similar_to
```

### Approach C: Knowledge Graph
```python
# Extract relationships as we enrich
# (Acme, competes_with, Beta), (Gamma, acquired, Delta)

enricher = TableEnricher(extract_relationships=True)
result.relationships  # List of (subject, predicate, object)
```

---

## Dimension 3: Knowledge Graph

**Extract entity relationships, build queryable graph.**

```
Acme ──competes_with──► Beta
  │                       │
  │ funded_by             │ acquired
  ▼                       ▼
Sequoia                 Gamma
```

**API:**
```python
enricher = TableEnricher(
    extract_relationships=True,
    relationship_types=["competes_with", "funded_by", "acquired", "partners_with"]
)

result = enricher.enrich(df, category="research")

result.graph.get_competitors("Acme")     # ["Beta", "Gamma"]
result.graph.path_between("Acme", "VC")  # Connection path
result.graph.to_networkx()               # For visualization
```

---

## Dimension 4: Intelligence Fusion

**LLM orchestrates multiple data sources.**

```
         ┌─────────────┐
         │   Web APIs  │ (Apollo, Crunchbase)
         └──────┬──────┘
                │
Row ───► LLM Orchestrator ───► Enriched Row
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
Web Search   Vector Store  Custom APIs
```

**API:**
```python
enricher = TableEnricher(
    sources=[
        WebSearchSource(api_key=tavily_key),
        CrunchbaseSource(api_key=cb_key),
        LLMKnowledgeSource(model="gpt-4o"),
    ]
)
```

---

## Dimension 5: Temporal Intelligence

**Track changes, detect staleness, delta updates.**

```
Day 1:  Acme Corp: funding = "Series B, $50M"
Day 30: Acme Corp: funding = "Series C, $120M" ← CHANGED
```

**API:**
```python
result = enricher.enrich_with_history(df, max_age_days=30)
result.changes  # Rows where values changed
result.stale    # Rows needing re-enrichment
```

---

## Engine Primitives (Generalizable)

The engine ships **primitives**, not domain-specific features.

### What's in the Engine

| Primitive | Description |
|-----------|-------------|
| **Context Injection** | Dataset summary in every prompt |
| **Entity Resolution** | Find duplicate/similar rows |
| **Relationship Extraction** | (subject, predicate, object) triples |
| **Row References** | Enrichment can reference other rows |
| **Aggregation Queries** | Questions about whole dataset |
| **Pairwise Comparison** | Compare two specific rows |

### What's NOT in the Engine (Product-Specific)

| Feature | Why Product-Specific |
|---------|---------------------|
| `funding_rank` | Assumes funding column |
| `cluster: "fintech_growth"` | Domain-specific |
| `score: 87` | Scoring logic varies |
| `recommended: true` | Criteria varies |

**Products compose primitives into domain intelligence.**

---

## Rich Output Schema

```python
class EnrichmentResult(BaseModel):
    data: DataFrame              # Original + enriched columns
    summary: EnrichmentSummary   # Stats
    context: Optional[str]       # Dataset summary
    graph: Optional[KnowledgeGraph]  # If enabled

class EnrichmentSummary(BaseModel):
    total_rows: int
    successful_rows: int
    cost: CostSummary
    duration_seconds: float
    cache_hit_rate: float
    avg_confidence: float

class CostSummary(BaseModel):
    total_usd: float
    breakdown: Dict[str, float]  # {"openai": 0.12, "tavily": 0.05}
    tokens_used: int
    cached_hits: int
```

---

## Engine Features for Products

### 1. Sources Column
Every field gets `{field}_sources` column with attribution.

### 2. Cost Tracking
```python
result.summary.cost.total_usd  # For billing
```

### 3. Caching (Phase 4 — COMPLETE)

SQLite-backed input-hash cache. Per-step-per-row granularity. Zero new dependencies (`sqlite3` is stdlib).

```python
# Enable caching for iterative development
pipeline = Pipeline([
    LLMStep("analyze", fields={...}),                                # Cached by default
    FunctionStep("search", fn=search, cache_version="v2"),           # Versioned caching
    LLMStep("synthesize", fields={...}, cache=False, depends_on=["search"]),  # Bypass cache
])
result = pipeline.run(df, config=EnrichmentConfig(
    enable_caching=True,
    cache_ttl=86400,          # 24 hours
    cache_dir=".lattice",     # cache.db location
))
# Cache stats in result
result.cost.steps["analyze"].cache_hit_rate  # 0.95
```

**Key design:** Cache key = `SHA256(step_name + row_data + prior_results + field_specs + model + temperature)`. Changing a field's prompt, enum, or type auto-invalidates — no manual cache busting needed. SQLite WAL mode for concurrent access. Checkpoint enhanced with `checkpoint_interval=100` for partial step progress on large datasets.

### 4. Streaming
```python
async for row_result in enricher.enrich_stream(df):
    yield row_result  # Update UI progressively
```

### 5. Hooks
```python
enricher = TableEnricher(
    hooks=EnrichmentHooks(
        on_row_complete=lambda r, e: update_ui(r, e),
        on_batch_complete=lambda df, cost: bill_customer(cost)
    )
)
```

---

## Eval System

### Types of Evals

| Eval | Description | When to Use |
|------|-------------|-------------|
| **Ground Truth** | Compare to known values | Have labeled data |
| **LLM-as-Judge** | Second LLM evaluates quality | No ground truth |
| **Consistency** | Same input → same output | Prompt tuning |
| **Source Verification** | Check cited sources | Trust building |
| **Freshness** | Data recency check | Temporal accuracy |

### API
```python
# Quick eval
result = enricher.eval(enriched_df, ground_truth=gt)
result.accuracy  # 0.85

# Full suite
report = enricher.eval_suite(df, enriched_df)
report.print()  # Pretty terminal output
```

### CI/CD Integration
```python
def test_enrichment_accuracy():
    result = enricher.eval(enriched, ground_truth=gt)
    assert result.accuracy >= 0.85
```

---

## Implementation Priority (Updated Feb 2026)

### Immediate: Quality + DX (Phases 3-4)
1. **API redesign** - `Pipeline.run(df)` as primary API, fields on steps, Enricher internal (Phase 2) ✅
2. **Resilience** - Per-row error handling, API retry/backoff (Phase 2) ✅
3. **Observability** - Progress reporting, cost aggregation (Phase 2) ✅
4. **Config/code cleanup** - Remove dead fields, dead code, FieldManager → utility (Phase 2) ✅
5. **Field spec + dynamic prompt** - 7-key field spec validation, dynamic prompt builder (markdown+XML), `default` enforcement, model default → mini (Phase 3) ✅
6. **Caching** - SQLite input-hash cache, per-step-per-row, TTL expiry, cache stats, checkpoint_interval, list[dict] input (Phase 4) ✅

### Ship: Minimum Viable Distribution (Phase 5A)
7. **Working examples** - 3-4 runnable scripts with sample data (including web search pattern)
8. **README rewrite** - Real examples, install instructions, quick start
9. **PyPI publish** - `pip install lattice-enrichment`

### Power User Features (Phase 5B)
10. **Lifecycle hooks** (#30) - Callbacks at pipeline/step/row boundaries for observability tools
11. **Prompt customization** (#34) - Three-tier: default, header injection, full override
12. **Web search utility** (#35) - Convenience function for common two-step pattern
13. **CLI** - `lattice run --csv data.csv --fields fields.csv`

### Differentiation (Post-launch)
14. **Provenance/sources** - Source attribution per field (#12)
15. **Cross-row analysis** - Dataset-aware enrichment
16. **Streaming output** - Progressive results (re-scope #11 for column-oriented model)

### Future Vision (unchanged)
12. **Relationship extraction** - Graph building
13. **Full knowledge graph** - Query, visualize
14. **Temporal intelligence** - Change detection

### Decisions Made (Feb 2026)
- **`Pipeline.run(df)` is the ONE way.** Enricher is internal. No FieldManager in public API.
- **Fields live on steps.** LLMStep accepts inline field specs. No separate field registry.
- **FieldManager → `load_fields()` utility.** CSV is a convenience, not a dependency.
- **No built-in web search.** FunctionStep is the escape hatch for any data source. Web search uses two-step pattern (FunctionStep → LLMStep) with citations as visible output.
- **LLMClient protocol + shipped adapters.** OpenAI default, Anthropic/Google as optional extras (~30 lines each). No litellm (too heavy).
- **No langfuse/eval tooling.** Expose hooks; users bring their own observability.
- **No waterfall resolution.** Source priority is user logic, not framework logic.
- **Evals are user-level.** Lattice exposes data. Users evaluate correctness.
- **Default model: gpt-4.1-mini.** Nano too limited for enrichment (no web search, hallucination risk with structured outputs). Mini matches GPT-4o at 83% cheaper. Users can override per-step.
- **7-key field spec.** `prompt`, `type`, `format`, `enum`, `examples`, `bad_examples`, `default`. Research-backed from Clay, Instructor, and OpenAI cookbook analysis.
- **Dynamic system prompt.** Markdown headers + XML data boundaries (OpenAI GPT-4.1 cookbook pattern). Only describes keys actually used. JSON in prompts avoided (performed poorly in OpenAI testing).
- **Three-tier prompt customization.** Default dynamic → `system_prompt_header=` injection → `system_prompt=` full override.

---

## Scale Limits & Positioning

**Sweet spot: 100 to 50,000 rows.** Primary use case: 1,000-10,000.

Lattice is single-process, in-memory (pandas), async. The bottleneck is almost always API rate limits and cost, not Lattice. With Tier 5 rate limits, Lattice processes 50K rows across 3 steps in under an hour.

**Users outgrow Lattice when:**
- **>50K-100K rows regularly** — need distributed processing (Spark, Ray, Prefect workers)
- **Real-time/streaming** — Lattice is batch-only
- **Multi-machine** — need distributed workers, not single-process asyncio
- **Complex orchestration** — conditional branching, loops, human-in-the-loop → Prefect/Dagster
- **Data exceeds memory** — >10GB DataFrames → Dask/Polars/Spark

This is intentional positioning: Lattice fills the gap between Instructor (single LLM call, 1-100 rows) and enterprise data infrastructure (Spark/Prefect/Airflow, 100K+ rows). Caching + checkpointing make the 1K-50K range viable for iterative development.

**Scale extensions (post-launch):**
- **Chunked execution** (`chunk_size=5000`): Process N rows at a time, cache carries across chunks. Extends ceiling to ~500K rows without going distributed.
- **`list[dict]` input** (Phase 4): Skip pandas conversion. Serves server contexts and Polars users.
- **pandas stays required**: 77% of data practitioners use pandas. Lattice's users are pandas users. No native Polars — `list[dict]` covers the gap.

## The Pitch Evolution

**Today (v0.4):**
> "Lattice is a composable pipeline engine for enriching DataFrames with LLMs"

**After Phase 2:**
> "...that handles the hard parts: retries, error recovery, cost tracking, and concurrency"

**After Phase 4:**
> "...with caching that auto-invalidates when you change prompts, and `list[dict]` input for any context"

**With Provenance (future):**
> "...and tells you exactly where each insight came from"

**With Graph (future):**
> "...and discovers relationships between entities"

---

*Internal planning document*
