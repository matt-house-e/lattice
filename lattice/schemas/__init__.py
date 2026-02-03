"""Pydantic schemas for structured LLM outputs.

This package provides type-safe schemas for LLM interactions:

- BaseLLMSchema: Base class with strict validation config
- StructuredResult[T]: Generic wrapper with usage tracking
- EnrichmentSpec: Field specification (what to extract)
- EnrichmentResult: Validated LLM output (what was extracted)

Example:
    from lattice.schemas import EnrichmentSpec, EnrichmentResult, StructuredResult

    # Define what to extract
    spec = EnrichmentSpec(
        field_name="market_size",
        prompt="Estimate the total addressable market",
        instructions="Provide in billions USD",
        data_type="String"
    )

    # Get validated result from LLM
    result: StructuredResult[EnrichmentResult] = await chain.acomplete_structured(...)

    # Access with type safety
    market_size = result.data.get("market_size")
    tokens_used = result.total_tokens
"""

from .base import BaseLLMSchema
from .structured import StructuredResult, UsageInfo
from .enrichment import EnrichmentSpec, EnrichmentResult

__all__ = [
    "BaseLLMSchema",
    "StructuredResult",
    "UsageInfo",
    "EnrichmentSpec",
    "EnrichmentResult",
]
