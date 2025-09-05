"""
Web-enriched LLM chain that combines web search with LLM processing.

Integrates Tavily web search API to provide real-time web context
for enhanced data enrichment capabilities.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union

from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from .llm import LLMChain
from ..core.exceptions import LLMError, ConfigurationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebEnrichedLLMChain(LLMChain):
    """
    Enhanced LLM chain that combines web search with LLM processing.
    
    Uses Tavily API to search the web for relevant information, then
    combines search results with row data for enriched LLM analysis.
    """
    
    def __init__(self, 
                 llm_chain: LLMChain,
                 tavily_api_key: Optional[str] = None,
                 max_search_results: int = 3,
                 search_depth: str = "basic"):
        """
        Initialize the Web-enriched LLM chain.
        
        Args:
            llm_chain: Base LLMChain to use for processing
            tavily_api_key: Tavily API key (uses environment variable if not provided)
            max_search_results: Maximum number of search results to include
            search_depth: Search depth ("basic" or "advanced")
        """
        # Initialize parent with the llm_chain's components
        super().__init__(llm_chain.llm, self._create_web_enhanced_prompt())
        
        # Set up Tavily client
        if tavily_api_key is None:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not tavily_api_key:
            raise ConfigurationError(
                "Tavily API key is required. Set TAVILY_API_KEY environment variable "
                "or pass tavily_api_key parameter to WebEnrichedLLMChain()"
            )
        
        try:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Tavily client: {e}")
        
        self.max_search_results = max_search_results
        self.search_depth = search_depth
        
    @classmethod
    def create(cls,
               api_key: Optional[str] = None,
               tavily_api_key: Optional[str] = None,
               model: str = "gpt-4o",
               temperature: float = 0.3,
               max_tokens: int = 8000,
               max_search_results: int = 5,
               search_depth: str = "advanced",
               **kwargs) -> 'WebEnrichedLLMChain':
        """
        Factory method to create a Web-enriched LLM chain.
        
        Args:
            api_key: OpenAI API key
            tavily_api_key: Tavily API key
            model: OpenAI model name
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            max_search_results: Maximum search results to include
            search_depth: Search depth ("basic" or "advanced")
            **kwargs: Additional arguments passed to LLMChain.openai()
            
        Returns:
            Configured WebEnrichedLLMChain instance
        """
        # Create base LLM chain
        llm_chain = LLMChain.openai(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Create web-enriched chain
        return cls(
            llm_chain=llm_chain,
            tavily_api_key=tavily_api_key,
            max_search_results=max_search_results,
            search_depth=search_depth
        )
    
    def _create_web_enhanced_prompt(self) -> ChatPromptTemplate:
        """Create prompt template that includes web search results."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a data enrichment specialist with access to real-time web information.

Company Data: {row_data}

Current Web Information:
{web_search_results}

Fields to analyze: {fields}

Using both the company data and the current web information, provide accurate, well-researched insights for each requested field.
Return your response as a JSON object with field names as keys and your analysis as values.

Important:
- Use the web information to provide current, factual responses
- Include citations or sources when referencing web information
- If web information contradicts company data, note the discrepancy
- Provide concise but comprehensive insights
- If web search yields no relevant results, rely on company data and general knowledge
- Format: {{"field_name": "your_analysis_with_sources"}}"""),
            ("user", "Analyze the company data using current web information and provide insights for the requested fields.")
        ])
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous invocation with web search enhancement.
        
        Args:
            input_data: Dictionary with 'row_data', 'fields', and optional 'category'
            
        Returns:
            Dictionary with 'output' key containing field values with web context
        """
        try:
            # Perform web search based on row data and fields
            search_results = self._perform_web_search(input_data)
            
            # Add search results to input data
            enhanced_input = input_data.copy()
            enhanced_input["web_search_results"] = search_results
            
            # Process through LLM with enhanced prompt including web search results
            try:
                # Format the prompt with web search results
                messages = self.prompt_template.format_messages(
                    row_data=enhanced_input.get("row_data", {}),
                    fields=enhanced_input.get("fields", {}),
                    web_search_results=enhanced_input.get("web_search_results", "No web search results available"),
                    category=enhanced_input.get("category", "")
                )
                
                # Get LLM response
                response = self.llm.invoke(messages)
                
                # Parse response
                output = self._parse_response(response.content, enhanced_input.get("fields", {}))
                
                return {"output": output}
                
            except Exception as llm_error:
                logger.error(f"LLM processing failed: {llm_error}")
                fields = enhanced_input.get("fields", {})
                return {"output": {field: "Error occurred" for field in fields.keys()}}
            
        except Exception as e:
            logger.error(f"WebEnrichedLLMChain invocation failed: {e}")
            # Fallback to LLM-only processing without web search
            # Fallback to LLM-only processing
            try:
                messages = self.prompt_template.format_messages(
                    row_data=input_data.get("row_data", {}),
                    fields=input_data.get("fields", {}),
                    web_search_results="Web search unavailable.",
                    category=input_data.get("category", "")
                )
                
                response = self.llm.invoke(messages)
                output = self._parse_response(response.content, input_data.get("fields", {}))
                return {"output": output}
                
            except Exception as fallback_error:
                logger.error(f"Fallback processing failed: {fallback_error}")
                fields = input_data.get("fields", {})
                return {"output": {field: "Error occurred" for field in fields.keys()}}
    
    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous invocation with web search enhancement.
        
        Args:
            input_data: Dictionary with 'row_data', 'fields', and optional 'category'
            
        Returns:
            Dictionary with 'output' key containing field values with web context
        """
        try:
            # Perform async web search
            search_results = await self._perform_web_search_async(input_data)
            
            # Add search results to input data
            enhanced_input = input_data.copy()
            enhanced_input["web_search_results"] = search_results
            
            # Process through LLM with enhanced prompt including web search results
            try:
                # Format the prompt with web search results
                messages = self.prompt_template.format_messages(
                    row_data=enhanced_input.get("row_data", {}),
                    fields=enhanced_input.get("fields", {}),
                    web_search_results=enhanced_input.get("web_search_results", "No web search results available"),
                    category=enhanced_input.get("category", "")
                )
                
                # Get LLM response asynchronously
                response = await self.llm.ainvoke(messages)
                
                # Parse response
                output = self._parse_response(response.content, enhanced_input.get("fields", {}))
                
                return {"output": output}
                
            except Exception as llm_error:
                logger.error(f"Async LLM processing failed: {llm_error}")
                fields = enhanced_input.get("fields", {})
                return {"output": {field: "Error occurred" for field in fields.keys()}}
            
        except Exception as e:
            logger.error(f"Async WebEnrichedLLMChain invocation failed: {e}")
            # Fallback to LLM-only processing
            try:
                messages = self.prompt_template.format_messages(
                    row_data=input_data.get("row_data", {}),
                    fields=input_data.get("fields", {}),
                    web_search_results="Web search unavailable.",
                    category=input_data.get("category", "")
                )
                
                response = await self.llm.ainvoke(messages)
                output = self._parse_response(response.content, input_data.get("fields", {}))
                return {"output": output}
                
            except Exception as fallback_error:
                logger.error(f"Async fallback processing failed: {fallback_error}")
                fields = input_data.get("fields", {})
                return {"output": {field: "Error occurred" for field in fields.keys()}}
    
    def _perform_web_search(self, input_data: Dict[str, Any]) -> str:
        """
        Perform web search based on input data and return formatted results.
        
        Args:
            input_data: Input data containing row_data and fields
            
        Returns:
            Formatted string containing search results
        """
        try:
            # Build search queries from row data and fields
            search_queries = self._build_search_queries(input_data)
            
            all_results = []
            
            for query in search_queries[:2]:  # Limit to 2 queries to manage API costs
                try:
                    response = self.tavily_client.search(
                        query=query,
                        max_results=self.max_search_results,
                        search_depth=self.search_depth,
                        include_answer=True,
                        include_raw_content=False
                    )
                    
                    # Process search results
                    if response and "results" in response:
                        formatted_results = self._format_search_results(response, query)
                        all_results.append(formatted_results)
                        
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    continue
            
            return "\n\n".join(all_results) if all_results else "No relevant web information found."
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "Web search encountered an error."
    
    async def _perform_web_search_async(self, input_data: Dict[str, Any]) -> str:
        """
        Asynchronously perform web search.
        
        For now, this is a wrapper around the sync version.
        In a full implementation, you'd want truly async Tavily operations.
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self._perform_web_search, input_data
        )
    
    def _build_search_queries(self, input_data: Dict[str, Any]) -> List[str]:
        """
        Build search queries from row data and field requirements.
        
        Args:
            input_data: Input data containing row_data and fields
            
        Returns:
            List of search query strings
        """
        row_data = input_data.get("row_data", {})
        fields = input_data.get("fields", {})
        
        queries = []
        
        # Extract company identifier for base query
        company_name = None
        for field in ["name", "company", "company_name", "organization"]:
            if field in row_data and row_data[field]:
                company_name = str(row_data[field]).strip()
                break
        
        if not company_name:
            # Fallback: use other identifying information
            identifiers = []
            for field in ["website", "domain", "url", "description"]:
                if field in row_data and row_data[field]:
                    identifiers.append(str(row_data[field]).strip())
            company_name = " ".join(identifiers[:2]) if identifiers else "company"
        
        # Build general company query
        base_query = f'"{company_name}"'
        
        # Add industry/sector context if available
        industry_info = []
        for field in ["industry", "sector", "category", "business_type"]:
            if field in row_data and row_data[field]:
                industry_info.append(str(row_data[field]).strip())
        
        if industry_info:
            base_query += f" {' '.join(industry_info[:2])}"
        
        queries.append(base_query)
        
        # Build field-specific queries
        for field_name, field_config in fields.items():
            if isinstance(field_config, dict) and "prompt" in field_config:
                # Extract key terms from field prompt for targeted search
                field_query = f'"{company_name}" {self._extract_search_terms_from_prompt(field_config["prompt"])}'
                queries.append(field_query)
        
        return queries[:3]  # Limit to 3 queries maximum
    
    def _extract_search_terms_from_prompt(self, prompt: str) -> str:
        """
        Extract key search terms from field prompt.
        
        Args:
            prompt: Field prompt text
            
        Returns:
            String of key search terms
        """
        # Simple keyword extraction - in production you might use NLP
        search_terms = []
        
        # Common business intelligence keywords
        keywords_map = {
            "funding": "funding investment capital",
            "revenue": "revenue earnings income",
            "market": "market size industry",
            "competition": "competitors competitive landscape",
            "leadership": "CEO founder executive team",
            "news": "news recent developments",
            "valuation": "valuation worth value",
            "growth": "growth expansion",
            "employees": "employees headcount team size"
        }
        
        prompt_lower = prompt.lower()
        for keyword, search_terms_for_keyword in keywords_map.items():
            if keyword in prompt_lower:
                search_terms.append(search_terms_for_keyword)
        
        return " ".join(search_terms[:2])  # Limit search terms
    
    def _format_search_results(self, response: Dict[str, Any], query: str) -> str:
        """
        Format Tavily search results for LLM consumption.
        
        Args:
            response: Tavily API response
            query: Original search query
            
        Returns:
            Formatted string containing search results
        """
        formatted = [f"Search Query: {query}\n"]
        
        # Include Tavily's answer if available
        if "answer" in response and response["answer"]:
            formatted.append(f"Summary: {response['answer']}\n")
        
        # Format individual results
        if "results" in response:
            formatted.append("Detailed Sources:")
            for i, result in enumerate(response["results"][:self.max_search_results], 1):
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                content = result.get("content", "No content")
                
                # Truncate content if too long
                if len(content) > 300:
                    content = content[:300] + "..."
                
                formatted.append(f"{i}. {title}")
                formatted.append(f"   Source: {url}")
                formatted.append(f"   Content: {content}\n")
        
        return "\n".join(formatted)


# Factory function for easy web-enriched chain creation
def create_web_enriched_chain(api_key: Optional[str] = None,
                            tavily_api_key: Optional[str] = None,
                            model: str = "gpt-4o",
                            **kwargs) -> WebEnrichedLLMChain:
    """
    Factory function to create a web-enriched LLM chain.
    
    Args:
        api_key: OpenAI API key
        tavily_api_key: Tavily API key  
        model: Model name
        **kwargs: Additional arguments
        
    Returns:
        Configured WebEnrichedLLMChain
    """
    return WebEnrichedLLMChain.create(
        api_key=api_key,
        tavily_api_key=tavily_api_key,
        model=model,
        **kwargs
    )