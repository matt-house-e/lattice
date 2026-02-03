"""Enrichment-specific Pydantic schemas.

Defines the contract between field specifications (what we ask the LLM)
and enrichment results (what the LLM returns, validated).
"""

from typing import Any, Literal, Optional
from pydantic import Field, field_validator, model_validator

from .base import BaseLLMSchema


class EnrichmentSpec(BaseLLMSchema):
    """Specification for a single enrichment field.

    Represents what we ask the LLM to produce for one field.
    Can be created from CSV row data or programmatically.

    V2 adds: output_format, quality_rules, sources, good/bad examples, fallback
    for improved LLM guidance and output quality.

    Attributes:
        field_name: The output column name
        prompt: What to extract/analyze (the instruction)
        output_format: Template for expected output structure
        quality_rules: What makes a good answer (replaces vague "instructions")
        sources: Preferred data sources for grounding
        data_type: Expected output type
        good_example: Example of ideal output (few-shot positive)
        bad_example: Example of what NOT to return (few-shot negative)
        fallback: What to return when data is insufficient
        examples: Legacy field for backwards compatibility
        instructions: Legacy field for backwards compatibility
    """

    field_name: str = Field(description="Output column name for this field")
    prompt: str = Field(description="What to extract or analyze")

    # V2 fields for improved guidance
    output_format: str = Field(default="", description="Template for output structure")
    quality_rules: str = Field(default="", description="What makes a good answer")
    sources: str = Field(default="", description="Preferred data sources")
    good_example: str = Field(default="", description="Example of ideal output")
    bad_example: str = Field(default="", description="Example of what NOT to return")
    fallback: str = Field(default="Unable to determine", description="Response when data insufficient")

    data_type: Literal["String", "Number", "Boolean", "Date", "JSON", "List"] = Field(
        default="String",
        description="Expected data type for validation"
    )

    # Legacy fields for backwards compatibility
    examples: list[str] = Field(default_factory=list, description="Legacy examples")
    instructions: str = Field(default="", description="Legacy instructions field")

    @field_validator('field_name')
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Ensure field name is a valid identifier."""
        if not v or not v.strip():
            raise ValueError("field_name cannot be empty")
        # Replace spaces with underscores for safety
        return v.strip().replace(' ', '_')

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Ensure prompt is not empty."""
        if not v or not v.strip():
            raise ValueError("prompt cannot be empty")
        return v.strip()

    def to_llm_spec(self) -> dict[str, Any]:
        """Convert to dictionary format for LLM prompt injection.

        Returns comprehensive spec including V2 fields when available,
        falling back to legacy fields for backwards compatibility.
        """
        spec: dict[str, Any] = {
            "prompt": self.prompt,
            "type": self.data_type
        }

        # V2 fields (preferred)
        if self.output_format:
            spec["output_format"] = self.output_format
        if self.quality_rules:
            spec["quality_rules"] = self.quality_rules
        if self.sources:
            spec["sources"] = self.sources
        if self.good_example:
            spec["good_example"] = self.good_example
        if self.bad_example:
            spec["bad_example"] = self.bad_example
        if self.fallback and self.fallback != "Unable to determine":
            spec["fallback"] = self.fallback

        # Legacy fields (backwards compatibility)
        if self.instructions and not self.quality_rules:
            spec["instructions"] = self.instructions
        if self.examples and not self.good_example:
            spec["examples"] = self.examples

        return spec


class EnrichmentResult(BaseLLMSchema):
    """Validated LLM response for enrichment.

    Contains the enriched field values returned by the LLM.
    Field names are dynamic (determined by EnrichmentSpec at runtime).

    The 'fields' dict maps field names to their values. Values can be:
    - str: For String type
    - int/float: For Number type
    - bool: For Boolean type
    - str (ISO format): For Date type
    - dict/list: For JSON type
    - list[str]: For List type

    Attributes:
        fields: Dictionary mapping field names to enriched values
    """

    # Allow extra fields since field names are dynamic
    model_config = BaseLLMSchema.model_config.copy()
    model_config['extra'] = 'allow'

    fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Enriched field values keyed by field name"
    )

    @model_validator(mode='before')
    @classmethod
    def capture_dynamic_fields(cls, data: Any) -> Any:
        """Capture top-level fields into the 'fields' dict.

        LLM returns {"market_size": "...", "competition_level": "..."}
        We capture these into fields dict for consistent access.
        """
        if isinstance(data, dict):
            # If 'fields' key exists, use it directly
            if 'fields' in data and isinstance(data['fields'], dict):
                return data

            # Otherwise, treat all keys as field values
            return {"fields": data}
        return data

    @model_validator(mode='after')
    def validate_no_null_strings(self) -> 'EnrichmentResult':
        """Convert 'null' strings and empty values to None.

        LLMs sometimes return the string "null" instead of actual null.
        Normalize these for consistent downstream handling.
        """
        for key, value in self.fields.items():
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned.lower() in ('null', 'none', 'n/a', ''):
                    self.fields[key] = None
                elif cleaned == "Unable to determine":
                    self.fields[key] = None
        return self

    def get(self, field_name: str, default: Any = None) -> Any:
        """Get a field value with optional default.

        Args:
            field_name: Name of the field to retrieve
            default: Value to return if field not found

        Returns:
            Field value or default
        """
        return self.fields.get(field_name, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dictionary for DataFrame integration.

        Returns:
            Dictionary of field names to values
        """
        return dict(self.fields)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: result['market_size']."""
        return self.fields[key]

    def __contains__(self, key: str) -> bool:
        """Allow 'in' checks: 'market_size' in result."""
        return key in self.fields
