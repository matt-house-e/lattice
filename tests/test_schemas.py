"""Tests for Pydantic schemas.

Tests validation, serialization, and LLM output handling for the schema classes.
"""

import pytest
from pydantic import ValidationError

from lattice.schemas import (
    BaseLLMSchema,
    StructuredResult,
    UsageInfo,
    EnrichmentSpec,
    EnrichmentResult,
)


class TestUsageInfo:
    """Tests for UsageInfo schema."""

    def test_basic_usage(self):
        """Test basic usage info creation."""
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_zero_tokens(self):
        """Test with zero tokens."""
        usage = UsageInfo(prompt_tokens=0, completion_tokens=0)
        assert usage.total_tokens == 0

    def test_negative_tokens_rejected(self):
        """Test that negative tokens are rejected."""
        with pytest.raises(ValidationError):
            UsageInfo(prompt_tokens=-1, completion_tokens=50)


class TestEnrichmentSpec:
    """Tests for EnrichmentSpec schema."""

    def test_basic_spec(self):
        """Test basic spec creation."""
        spec = EnrichmentSpec(
            field_name="market_size",
            prompt="Estimate the market size",
            instructions="Provide in billions USD",
            data_type="String"
        )
        assert spec.field_name == "market_size"
        assert spec.prompt == "Estimate the market size"
        assert spec.data_type == "String"

    def test_field_name_with_spaces(self):
        """Test that spaces in field names are converted to underscores."""
        spec = EnrichmentSpec(
            field_name="market size estimate",
            prompt="Test prompt"
        )
        assert spec.field_name == "market_size_estimate"

    def test_empty_field_name_rejected(self):
        """Test that empty field names are rejected."""
        with pytest.raises(ValidationError):
            EnrichmentSpec(field_name="", prompt="Test")

    def test_empty_prompt_rejected(self):
        """Test that empty prompts are rejected."""
        with pytest.raises(ValidationError):
            EnrichmentSpec(field_name="test", prompt="")

    def test_default_data_type(self):
        """Test default data type is String."""
        spec = EnrichmentSpec(field_name="test", prompt="Test prompt")
        assert spec.data_type == "String"

    def test_with_examples(self):
        """Test spec with examples."""
        spec = EnrichmentSpec(
            field_name="market_size",
            prompt="Estimate the market size",
            examples=["$50B", "$100B"]
        )
        assert len(spec.examples) == 2
        assert "$50B" in spec.examples

    def test_to_llm_spec(self):
        """Test conversion to LLM-ready dict."""
        spec = EnrichmentSpec(
            field_name="market_size",
            prompt="Estimate the market",
            instructions="In billions",
            data_type="String",
            examples=["$50B"]
        )
        llm_spec = spec.to_llm_spec()
        assert llm_spec["prompt"] == "Estimate the market"
        assert llm_spec["instructions"] == "In billions"
        assert llm_spec["type"] == "String"
        assert llm_spec["examples"] == ["$50B"]


class TestEnrichmentResult:
    """Tests for EnrichmentResult schema."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = EnrichmentResult(fields={"market_size": "$50B"})
        assert result.fields["market_size"] == "$50B"

    def test_dynamic_fields_capture(self):
        """Test that top-level fields are captured into 'fields' dict."""
        # This simulates what the LLM returns
        data = {"market_size": "$50B", "competition_level": "High"}
        result = EnrichmentResult.model_validate(data)
        assert result.fields["market_size"] == "$50B"
        assert result.fields["competition_level"] == "High"

    def test_null_string_normalization(self):
        """Test that 'null' strings are converted to None."""
        result = EnrichmentResult(fields={"test": "null"})
        assert result.fields["test"] is None

    def test_unable_to_determine_normalization(self):
        """Test that 'Unable to determine' is converted to None."""
        result = EnrichmentResult(fields={"test": "Unable to determine"})
        assert result.fields["test"] is None

    def test_get_method(self):
        """Test get() with default."""
        result = EnrichmentResult(fields={"a": "1"})
        assert result.get("a") == "1"
        assert result.get("b", "default") == "default"

    def test_dict_access(self):
        """Test dict-like access."""
        result = EnrichmentResult(fields={"a": "1"})
        assert result["a"] == "1"

    def test_contains(self):
        """Test 'in' operator."""
        result = EnrichmentResult(fields={"a": "1"})
        assert "a" in result
        assert "b" not in result

    def test_to_dict(self):
        """Test conversion to plain dict."""
        result = EnrichmentResult(fields={"a": "1", "b": "2"})
        d = result.to_dict()
        assert d == {"a": "1", "b": "2"}


class TestStructuredResult:
    """Tests for StructuredResult generic wrapper."""

    def test_basic_structured_result(self):
        """Test basic structured result creation."""
        data = EnrichmentResult(fields={"test": "value"})
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50)

        result = StructuredResult[EnrichmentResult](
            data=data,
            usage=usage,
            full_prompt="Test prompt"
        )

        assert result.data.fields["test"] == "value"
        assert result.total_tokens == 150
        assert result.full_prompt == "Test prompt"

    def test_convenience_properties(self):
        """Test convenience token properties."""
        data = EnrichmentResult(fields={})
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50)
        result = StructuredResult[EnrichmentResult](data=data, usage=usage)

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150


class TestBaseLLMSchema:
    """Tests for BaseLLMSchema base class."""

    def test_extra_forbid(self):
        """Test that extra fields are rejected."""

        class TestSchema(BaseLLMSchema):
            name: str

        with pytest.raises(ValidationError):
            TestSchema(name="test", extra_field="invalid")

    def test_enum_values(self):
        """Test that enums serialize to values."""
        from enum import Enum

        class Status(str, Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class TestSchema(BaseLLMSchema):
            status: Status

        schema = TestSchema(status=Status.ACTIVE)
        # Should serialize enum to value
        assert schema.model_dump()["status"] == "active"
