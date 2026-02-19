"""Enrichment schema types for the Lattice pipeline."""

from pydantic import BaseModel, ConfigDict


class EnrichmentResult(BaseModel):
    """Default validation schema for LLM enrichment results.

    Allows arbitrary fields so any JSON dict from the LLM passes validation.
    Users can supply a stricter schema to LLMStep for typed validation.
    """

    model_config = ConfigDict(extra="allow")
