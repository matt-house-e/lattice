"""Integration test for Pydantic structured outputs.

Tests the full flow: CSV → EnrichmentSpec → LLM mock → EnrichmentResult validation.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
import json

from lattice.data.fields import FieldManager
from lattice.chains.llm import LLMChain
from lattice.schemas import EnrichmentSpec, EnrichmentResult, StructuredResult, UsageInfo


class TestStructuredIntegration:
    """Integration tests for structured output pipeline."""

    def test_csv_to_enrichment_specs(self):
        """Test CSV loading produces valid EnrichmentSpec objects."""
        fm = FieldManager("examples/field_categories.csv")

        # Get specs for business_analysis category
        specs = fm.get_enrichment_specs("business_analysis")

        assert len(specs) == 3
        assert all(isinstance(s, EnrichmentSpec) for s in specs)

        # Check field names
        field_names = [s.field_name for s in specs]
        assert "market_size" in field_names
        assert "competition_level" in field_names
        assert "growth_potential" in field_names

    def test_specs_to_llm_dict(self):
        """Test EnrichmentSpec converts to LLM-ready dict."""
        fm = FieldManager("examples/field_categories.csv")
        specs_dict = fm.get_specs_as_dict("business_analysis")

        assert "market_size" in specs_dict
        assert "prompt" in specs_dict["market_size"]
        assert "type" in specs_dict["market_size"]
        assert "instructions" in specs_dict["market_size"]

    def test_llm_response_validation(self):
        """Test LLM response validates correctly."""
        # Simulate what LLM returns
        llm_output = {
            "market_size": "$50B - Cloud infrastructure",
            "competition_level": "High",
            "growth_potential": "Medium"
        }

        result = EnrichmentResult.model_validate(llm_output)

        assert result["market_size"] == "$50B - Cloud infrastructure"
        assert "market_size" in result
        assert result.to_dict() == llm_output

    def test_validation_error_feedback(self):
        """Test that validation errors can be serialized for LLM feedback."""
        from pydantic import ValidationError

        # EnrichmentSpec requires non-empty prompt
        with pytest.raises(ValidationError) as exc_info:
            EnrichmentSpec(field_name="test", prompt="")

        # Error should be JSON-serializable for LLM feedback
        error_json = exc_info.value.json()
        assert "prompt" in error_json.lower()

    def test_structured_result_wrapper(self):
        """Test StructuredResult wraps data with usage info."""
        data = EnrichmentResult(fields={"test": "value"})
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50)

        result = StructuredResult[EnrichmentResult](
            data=data,
            usage=usage,
            full_prompt="Test prompt"
        )

        assert result.data["test"] == "value"
        assert result.total_tokens == 150
        assert result.full_prompt == "Test prompt"

    def test_complete_structured_mock(self):
        """Test complete_structured with mocked LLM."""
        # Create chain with mocked LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "fields": {
                "market_size": "$50B",
                "competition": "High"
            }
        })
        mock_response.usage_metadata = MagicMock(input_tokens=100, output_tokens=50)
        mock_llm.invoke.return_value = mock_response

        chain = LLMChain(mock_llm)

        # Call complete_structured
        result = chain.complete_structured(
            {"row_data": {"company": "TestCo"}, "fields": {}},
            schema=EnrichmentResult
        )

        assert isinstance(result, StructuredResult)
        assert result.data["market_size"] == "$50B"
        assert mock_llm.invoke.called

    def test_acomplete_structured_mock(self):
        """Test async complete_structured with mocked LLM."""
        import asyncio

        async def run_test():
            # Create chain with mocked async LLM
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = json.dumps({
                "fields": {"result": "async works"}
            })
            mock_response.usage_metadata = None
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)

            chain = LLMChain(mock_llm)

            result = await chain.acomplete_structured(
                {"row_data": {}, "fields": {}},
                schema=EnrichmentResult
            )

            assert result.data["result"] == "async works"
            assert mock_llm.ainvoke.called

        asyncio.get_event_loop().run_until_complete(run_test())
