"""Enrichment schema types for the Lattice pipeline."""

from pydantic import BaseModel, ConfigDict


class EnrichmentSpec(BaseModel):
    """Specification for a single enrichment field, loaded from CSV."""

    prompt: str
    instructions: str = ""
    data_type: str = "String"
    examples: list[str] = []
    output_format: str = ""
    quality_rules: str = ""
    sources: str = ""
    good_example: str = ""
    bad_example: str = ""
    fallback: str = ""


class EnrichmentResult(BaseModel):
    """Default validation schema for LLM enrichment results.

    Allows arbitrary fields so any JSON dict from the LLM passes validation.
    Users can supply a stricter schema to LLMStep for typed validation.
    """

    model_config = ConfigDict(extra="allow")
