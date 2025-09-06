"""
Unit tests for JSON parsing functionality in LLM chains.

Tests the enhanced JSON parsing that handles markdown-wrapped responses
and provides graceful fallbacks for unparseable content.
"""

import unittest
import json
from lattice.chains.llm import LLMChain


class TestJSONParsing(unittest.TestCase):
    """Test cases for JSON response parsing in LLM chains."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal chain instance for testing parsing methods
        self.chain = LLMChain.__new__(LLMChain)
        
        # Test fields dictionary
        self.test_fields = {
            "business_impact": {"prompt": "Analyze business impact"},
            "technical_domain": {"prompt": "Identify technical domain"},
            "complexity_level": {"prompt": "Assess complexity"},
            "effort_estimate": {"prompt": "Estimate effort"}
        }
    
    def test_markdown_wrapped_json(self):
        """Test parsing of JSON wrapped in markdown code blocks."""
        response = '''```json
{
  "business_impact": "High",
  "technical_domain": "Software Applications",
  "complexity_level": "Moderate", 
  "effort_estimate": "2-5 days"
}
```'''
        
        result = self.chain._parse_response(response, self.test_fields)
        
        self.assertEqual(result["business_impact"], "High")
        self.assertEqual(result["technical_domain"], "Software Applications")
        self.assertEqual(result["complexity_level"], "Moderate")
        self.assertEqual(result["effort_estimate"], "2-5 days")
    
    def test_raw_json(self):
        """Test parsing of raw JSON without markdown."""
        response = '''{
  "business_impact": "Critical",
  "technical_domain": "Infrastructure", 
  "complexity_level": "Expert",
  "effort_estimate": "1+ weeks"
}'''
        
        result = self.chain._parse_response(response, self.test_fields)
        
        self.assertEqual(result["business_impact"], "Critical")
        self.assertEqual(result["technical_domain"], "Infrastructure")
        self.assertEqual(result["complexity_level"], "Expert")
        self.assertEqual(result["effort_estimate"], "1+ weeks")
    
    def test_mixed_format_with_explanation(self):
        """Test parsing JSON from responses with additional explanation."""
        response = '''Based on the analysis, here are the field values:

```json
{
  "business_impact": "Medium",
  "technical_domain": "Network",
  "complexity_level": "Simple",
  "effort_estimate": "1-4 hours"
}
```

This assessment is based on the standard network configuration requirements.'''
        
        result = self.chain._parse_response(response, self.test_fields)
        
        self.assertEqual(result["business_impact"], "Medium")
        self.assertEqual(result["technical_domain"], "Network")
        self.assertEqual(result["complexity_level"], "Simple")
        self.assertEqual(result["effort_estimate"], "1-4 hours")
    
    def test_text_parsing_fallback(self):
        """Test fallback text parsing when JSON extraction fails."""
        response = '''The business_impact is "High" and the technical_domain should be "Security". 
        For complexity_level, I'd rate this as "Complex" and effort_estimate would be "2-5 days".'''
        
        result = self.chain._parse_response(response, self.test_fields)
        
        self.assertEqual(result["business_impact"], "High")
        self.assertEqual(result["technical_domain"], "Security")
        self.assertEqual(result["complexity_level"], "Complex")
        self.assertEqual(result["effort_estimate"], "2-5 days")
    
    def test_graceful_fallback(self):
        """Test graceful fallback for completely unparseable content."""
        response = "This is just random text with no structured data whatsoever."
        
        result = self.chain._parse_response(response, self.test_fields)
        
        # Should return default fallback for all fields
        for field in self.test_fields:
            self.assertEqual(result[field], "Unable to parse response")
    
    def test_no_duplicate_content(self):
        """Test that the fix prevents content duplication across fields."""
        # This was the original bug - long responses getting duplicated
        response = '''```json
{
  "business_impact": "High",
  "technical_domain": "Infrastructure"  
}
```

Additional explanation that used to get duplicated across all fields.'''
        
        result = self.chain._parse_response(response, self.test_fields)
        
        # Verify clean, individual field values (no duplication)
        self.assertEqual(result["business_impact"], "High")
        self.assertEqual(result["technical_domain"], "Infrastructure")
        
        # When JSON is successfully parsed, we only get the parsed values
        # (This is actually correct behavior - the processor will handle missing fields)
        self.assertEqual(len(result), 2)  # Only the two fields from the JSON
        
        # Most importantly, no field should contain the entire response text
        for field, value in result.items():
            self.assertLess(len(str(value)), 200, 
                          f"Field {field} contains unexpectedly long content: {value}")
            # Values should be clean, not contain markdown or extra text
            self.assertNotIn("```", str(value))
            self.assertNotIn("Additional explanation", str(value))


if __name__ == "__main__":
    unittest.main()