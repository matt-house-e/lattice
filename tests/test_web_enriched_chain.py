#!/usr/bin/env python3
"""
Test script for WebEnrichedLLMChain functionality.

Tests web search integration with LLM processing for enhanced data enrichment.
"""

import os
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import using the new package structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice.chains.web_enriched import WebEnrichedLLMChain, create_web_enriched_chain
from lattice.chains.llm import LLMChain
from lattice.core.exceptions import ConfigurationError


class TestWebEnrichedLLMChain:
    """Test suite for WebEnrichedLLMChain functionality."""
    
    @pytest.fixture
    def mock_tavily_response(self):
        """Mock Tavily API response."""
        return {
            "results": [
                {
                    "title": "Company ABC - Latest Funding Round",
                    "url": "https://example.com/news/company-abc-funding",
                    "content": "Company ABC just raised $50M in Series B funding led by XYZ Ventures. The company plans to use the funds for market expansion and product development."
                },
                {
                    "title": "Company ABC - Market Analysis",
                    "url": "https://example.com/analysis/company-abc-market",
                    "content": "Company ABC operates in the $10B cloud infrastructure market, competing with major players like AWS and Azure."
                }
            ],
            "answer": "Company ABC is a cloud infrastructure company that recently raised $50M in Series B funding."
        }
    
    @pytest.fixture
    def sample_row_data(self):
        """Sample company data for testing."""
        return {
            "name": "Company ABC",
            "industry": "Cloud Infrastructure", 
            "website": "https://companyabc.com",
            "description": "Leading cloud infrastructure provider"
        }
    
    @pytest.fixture
    def sample_fields(self):
        """Sample field configuration for testing."""
        return {
            "funding_status": {
                "prompt": "What is the latest funding status of this company?",
                "type": "String",
                "instructions": "Provide recent funding information"
            },
            "market_position": {
                "prompt": "What is this company's position in the market?",
                "type": "String", 
                "instructions": "Analyze competitive positioning"
            }
        }
    
    def test_web_enriched_chain_initialization_success(self):
        """Test successful initialization of WebEnrichedLLMChain."""
        # Mock the base LLM chain
        mock_llm_chain = Mock(spec=LLMChain)
        mock_llm_chain.llm = Mock()
        
        with patch('lattice.chains.web_enriched.TavilyClient') as mock_tavily:
            mock_tavily.return_value = Mock()
            
            chain = WebEnrichedLLMChain(
                llm_chain=mock_llm_chain,
                tavily_api_key="test-key",
                max_search_results=3
            )
            
            assert chain.max_search_results == 3
            assert chain.search_depth == "basic"
            mock_tavily.assert_called_once_with(api_key="test-key")
    
    def test_web_enriched_chain_missing_api_key(self):
        """Test initialization fails with missing Tavily API key."""
        mock_llm_chain = Mock(spec=LLMChain)
        mock_llm_chain.llm = Mock()
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="Tavily API key is required"):
                WebEnrichedLLMChain(llm_chain=mock_llm_chain)
    
    def test_factory_method_create(self):
        """Test the factory method for creating WebEnrichedLLMChain."""
        with patch('lattice.chains.web_enriched.TavilyClient') as mock_tavily, \
             patch('lattice.chains.llm.LLMChain.openai') as mock_openai:
            
            mock_tavily.return_value = Mock()
            mock_openai.return_value = Mock(spec=LLMChain)
            mock_openai.return_value.llm = Mock()
            
            chain = WebEnrichedLLMChain.create(
                api_key="test-openai-key",
                tavily_api_key="test-tavily-key",
                model="gpt-4",
                temperature=0.7
            )
            
            assert isinstance(chain, WebEnrichedLLMChain)
            mock_openai.assert_called_once()
            mock_tavily.assert_called_once_with(api_key="test-tavily-key")
    
    def test_build_search_queries(self, sample_row_data, sample_fields):
        """Test search query building from row data and fields."""
        mock_llm_chain = Mock(spec=LLMChain)
        mock_llm_chain.llm = Mock()
        
        with patch('lattice.chains.web_enriched.TavilyClient') as mock_tavily:
            mock_tavily.return_value = Mock()
            
            chain = WebEnrichedLLMChain(
                llm_chain=mock_llm_chain,
                tavily_api_key="test-key"
            )
            
            input_data = {
                "row_data": sample_row_data,
                "fields": sample_fields
            }
            
            queries = chain._build_search_queries(input_data)
            
            assert len(queries) > 0
            assert any("Company ABC" in query for query in queries)
            assert any("Cloud Infrastructure" in query for query in queries)
    
    def test_extract_search_terms_from_prompt(self):
        """Test extraction of search terms from field prompts."""
        mock_llm_chain = Mock(spec=LLMChain)
        mock_llm_chain.llm = Mock()
        
        with patch('lattice.chains.web_enriched.TavilyClient') as mock_tavily:
            mock_tavily.return_value = Mock()
            
            chain = WebEnrichedLLMChain(
                llm_chain=mock_llm_chain,
                tavily_api_key="test-key"
            )
            
            # Test funding prompt
            funding_prompt = "What is the latest funding status and investment rounds?"
            terms = chain._extract_search_terms_from_prompt(funding_prompt)
            assert "funding" in terms or "investment" in terms
            
            # Test revenue prompt
            revenue_prompt = "Analyze the company's revenue and earnings performance"
            terms = chain._extract_search_terms_from_prompt(revenue_prompt)
            assert "revenue" in terms or "earnings" in terms
    
    def test_format_search_results(self, mock_tavily_response):
        """Test formatting of Tavily search results."""
        mock_llm_chain = Mock(spec=LLMChain)
        mock_llm_chain.llm = Mock()
        
        with patch('lattice.chains.web_enriched.TavilyClient') as mock_tavily:
            mock_tavily.return_value = Mock()
            
            chain = WebEnrichedLLMChain(
                llm_chain=mock_llm_chain,
                tavily_api_key="test-key"
            )
            
            formatted = chain._format_search_results(mock_tavily_response, "test query")
            
            assert "test query" in formatted
            assert "Company ABC" in formatted
            assert "$50M" in formatted
            assert "https://example.com" in formatted
    
    @patch('lattice.chains.web_enriched.TavilyClient')
    def test_perform_web_search_success(self, mock_tavily_client, sample_row_data, 
                                        sample_fields, mock_tavily_response):
        """Test successful web search execution."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_tavily_response
        mock_tavily_client.return_value = mock_client_instance
        
        mock_llm_chain = Mock(spec=LLMChain)
        mock_llm_chain.llm = Mock()
        
        chain = WebEnrichedLLMChain(
            llm_chain=mock_llm_chain,
            tavily_api_key="test-key"
        )
        
        input_data = {
            "row_data": sample_row_data,
            "fields": sample_fields
        }
        
        result = chain._perform_web_search(input_data)
        
        assert "Company ABC" in result
        assert "$50M" in result
        mock_client_instance.search.assert_called()
    
    @patch('lattice.chains.web_enriched.TavilyClient')
    def test_perform_web_search_failure(self, mock_tavily_client, sample_row_data, sample_fields):
        """Test web search handling when Tavily API fails."""
        # Setup mock to raise exception
        mock_client_instance = Mock()
        mock_client_instance.search.side_effect = Exception("API Error")
        mock_tavily_client.return_value = mock_client_instance
        
        mock_llm_chain = Mock(spec=LLMChain)
        mock_llm_chain.llm = Mock()
        
        chain = WebEnrichedLLMChain(
            llm_chain=mock_llm_chain,
            tavily_api_key="test-key"
        )
        
        input_data = {
            "row_data": sample_row_data,
            "fields": sample_fields
        }
        
        result = chain._perform_web_search(input_data)
        
        assert "No relevant web information found" in result or "Web search encountered an error" in result
    
    @patch('lattice.chains.web_enriched.TavilyClient')
    def test_invoke_with_web_search(self, mock_tavily_client, sample_row_data, 
                                    sample_fields, mock_tavily_response):
        """Test full invoke method with web search integration."""
        # Setup Tavily mock
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = mock_tavily_response
        mock_tavily_client.return_value = mock_client_instance
        
        # Setup LLM chain mock
        mock_llm_chain = Mock(spec=LLMChain)
        mock_llm_chain.llm = Mock()
        mock_llm_chain.invoke.return_value = {
            "output": {
                "funding_status": "Recently raised $50M in Series B",
                "market_position": "Strong position in cloud infrastructure"
            }
        }
        
        # Create and test the chain
        chain = WebEnrichedLLMChain(
            llm_chain=mock_llm_chain,
            tavily_api_key="test-key"
        )
        
        input_data = {
            "row_data": sample_row_data,
            "fields": sample_fields
        }
        
        result = chain.invoke(input_data)
        
        # Verify results
        assert "output" in result
        assert "funding_status" in result["output"]
        assert "market_position" in result["output"]
        
        # Verify LLM chain was called with enhanced input including web search results
        mock_llm_chain.invoke.assert_called_once()
        call_args = mock_llm_chain.invoke.call_args[0][0]
        assert "web_search_results" in call_args
        assert "Company ABC" in call_args["web_search_results"]
    
    @patch('lattice.chains.web_enriched.TavilyClient')
    def test_invoke_fallback_on_search_failure(self, mock_tavily_client, sample_row_data, sample_fields):
        """Test fallback to LLM-only processing when web search fails."""
        # Setup Tavily mock to fail
        mock_client_instance = Mock()
        mock_client_instance.search.side_effect = Exception("Search failed")
        mock_tavily_client.return_value = mock_client_instance
        
        # Setup LLM chain mock
        mock_llm_chain = Mock(spec=LLMChain)
        mock_llm_chain.llm = Mock()
        mock_llm_chain.invoke.return_value = {
            "output": {
                "funding_status": "Unable to determine from available data",
                "market_position": "Analysis based on company data only"
            }
        }
        
        chain = WebEnrichedLLMChain(
            llm_chain=mock_llm_chain,
            tavily_api_key="test-key"
        )
        
        input_data = {
            "row_data": sample_row_data,
            "fields": sample_fields
        }
        
        result = chain.invoke(input_data)
        
        # Should still return results from LLM fallback
        assert "output" in result
        mock_llm_chain.invoke.assert_called_once()
        
        # Verify it was called with search failure message
        call_args = mock_llm_chain.invoke.call_args[0][0]
        assert "web_search_results" in call_args
        assert "Web search unavailable" in call_args["web_search_results"]
    
    def test_factory_function(self):
        """Test the standalone factory function."""
        with patch('lattice.chains.web_enriched.WebEnrichedLLMChain.create') as mock_create:
            mock_create.return_value = Mock()
            
            result = create_web_enriched_chain(
                api_key="test-openai",
                tavily_api_key="test-tavily"
            )
            
            mock_create.assert_called_once_with(
                api_key="test-openai",
                tavily_api_key="test-tavily",
                model="gpt-3.5-turbo"
            )


class TestWebEnrichedChainIntegration:
    """Integration tests for WebEnrichedLLMChain with real components."""
    
    def test_integration_with_mock_responses(self):
        """Test integration with mocked external dependencies."""
        # Skip if no API keys available
        openai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY") or "test-key"
        
        if not openai_key or openai_key == "your-api-key-here":
            pytest.skip("OpenAI API key not available for integration test")
        
        with patch('lattice.chains.web_enriched.TavilyClient') as mock_tavily:
            # Mock successful Tavily response
            mock_client = Mock()
            mock_client.search.return_value = {
                "results": [
                    {
                        "title": "Test Company News",
                        "url": "https://example.com/news",
                        "content": "Test company raised $10M funding recently."
                    }
                ],
                "answer": "Test company is a technology startup."
            }
            mock_tavily.return_value = mock_client
            
            # Create chain with real OpenAI but mocked Tavily
            try:
                chain = WebEnrichedLLMChain.create(
                    api_key=openai_key,
                    tavily_api_key=tavily_key,
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=1000
                )
                
                # Test data
                input_data = {
                    "row_data": {
                        "name": "Test Company",
                        "industry": "Technology"
                    },
                    "fields": {
                        "funding_status": {
                            "prompt": "What is the funding status?",
                            "type": "String"
                        }
                    }
                }
                
                # Should not raise exceptions
                result = chain.invoke(input_data)
                assert "output" in result
                assert "funding_status" in result["output"]
                
                # Verify web search was called
                mock_client.search.assert_called()
                
            except ConfigurationError:
                pytest.skip("Configuration error in integration test")


def run_manual_test():
    """Manual test function for development and debugging."""
    print("🔍 Testing WebEnrichedLLMChain")
    print("=" * 50)
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key or openai_key == "your-api-key-here":
        print("⚠️  OpenAI API key not set. Using mocked LLM responses.")
    if not tavily_key:
        print("⚠️  Tavily API key not set. Using mocked search responses.")
    
    try:
        # Test with mock data
        sample_data = {
            "row_data": {
                "name": "OpenAI",
                "industry": "Artificial Intelligence",
                "website": "https://openai.com"
            },
            "fields": {
                "recent_funding": {
                    "prompt": "What is the latest funding information for this company?",
                    "type": "String"
                },
                "market_position": {
                    "prompt": "What is this company's position in the AI market?", 
                    "type": "String"
                }
            }
        }
        
        with patch('lattice.chains.web_enriched.TavilyClient') as mock_tavily:
            # Mock Tavily for testing
            mock_client = Mock()
            mock_client.search.return_value = {
                "results": [
                    {
                        "title": "OpenAI Funding News",
                        "url": "https://example.com/openai-funding",
                        "content": "OpenAI has raised significant funding from Microsoft and other investors, valuing the company at $80+ billion."
                    }
                ],
                "answer": "OpenAI is a leading AI company with major backing from Microsoft."
            }
            mock_tavily.return_value = mock_client
            
            # Create and test chain
            chain = WebEnrichedLLMChain.create(
                api_key=openai_key or "mock-key",
                tavily_api_key=tavily_key or "mock-key"
            )
            
            print("✅ Chain created successfully")
            
            # Test search query building
            queries = chain._build_search_queries(sample_data)
            print(f"📝 Generated search queries: {queries}")
            
            # Test search results formatting
            mock_response = {
                "results": [{"title": "Test", "url": "https://test.com", "content": "Test content"}],
                "answer": "Test answer"
            }
            formatted = chain._format_search_results(mock_response, "test query")
            print(f"📄 Formatted search results preview: {formatted[:200]}...")
            
            print("\n✅ All tests passed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run manual tests when script is executed directly
    run_manual_test()