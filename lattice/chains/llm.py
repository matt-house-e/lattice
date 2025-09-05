"""
Chain classes for the Lattice enrichment tool.

Simple, focused wrappers around LangChain functionality with clean APIs.
Provides both LLM-only chains and vector store enhanced chains.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..core.exceptions import LLMError, ConfigurationError

logger = logging.getLogger(__name__)


class BaseLLMChain(ABC):
    """Abstract base class for LLM chains."""
    
    @abstractmethod
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous invocation of the chain."""
        pass
    
    @abstractmethod
    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronous invocation of the chain."""
        pass


class LLMChain(BaseLLMChain):
    """
    Simple LangChain wrapper for LLM-based enrichment.
    
    Provides a clean interface for LLM enrichment with proper error handling,
    response parsing, and both sync/async support.
    """
    
    def __init__(self, 
                 llm: BaseLanguageModel, 
                 prompt_template: Optional[ChatPromptTemplate] = None):
        """
        Initialize the LLM chain.
        
        Args:
            llm: The language model to use
            prompt_template: Custom prompt template (optional, uses default if not provided)
        """
        self.llm = llm
        self.prompt_template = prompt_template or self._create_default_prompt()
    
    @classmethod
    def openai(cls, 
               model: str = "gpt-4o-mini",
               api_key: Optional[str] = None,
               temperature: float = 0.5,
               max_tokens: int = 8000,
               **kwargs) -> 'LLMChain':
        """
        Factory method to create an OpenAI-based LLM chain.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (uses environment variable if not provided)
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments passed to ChatOpenAI
            
        Returns:
            Configured LLMChain instance
            
        Raises:
            ConfigurationError: If API key is missing or invalid
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key or api_key == "your-api-key-here":
            raise ConfigurationError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter to LLMChain.openai()"
            )
        
        try:
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return cls(llm)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI LLM: {e}")
    
    def _create_default_prompt(self) -> ChatPromptTemplate:
        """Create the default prompt template for enrichment."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a data enrichment specialist. Given company data and specific fields to analyze, 
provide accurate, concise insights for each requested field.

Company Data: {row_data}

Fields to analyze: {fields}

For each field, provide a response that matches the field's requirements and data type.
Return your response as a JSON object with field names as keys and your analysis as values.

Important: 
- Provide factual, well-reasoned responses
- Match the expected data type for each field
- If you cannot determine a value, use "Unable to determine" rather than guessing
- Keep responses concise but informative"""),
            ("user", "Analyze the company data and provide insights for the requested fields.")
        ])
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous invocation of the LLM chain.
        
        Args:
            input_data: Dictionary with 'row_data', 'fields', and optional 'category'
            
        Returns:
            Dictionary with 'output' key containing field values
            
        Raises:
            LLMError: If LLM invocation fails
        """
        try:
            # Format the prompt
            messages = self.prompt_template.format_messages(
                row_data=input_data.get("row_data", {}),
                fields=input_data.get("fields", {}),
                category=input_data.get("category", "")
            )
            
            # Get LLM response
            response = self.llm.invoke(messages)
            
            # Parse response
            output = self._parse_response(response.content, input_data.get("fields", {}))
            
            return {"output": output}
            
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            # Return empty values for all fields on error
            fields = input_data.get("fields", {})
            return {"output": {field: "Error occurred" for field in fields.keys()}}
    
    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous invocation of the LLM chain.
        
        Args:
            input_data: Dictionary with 'row_data', 'fields', and optional 'category'
            
        Returns:
            Dictionary with 'output' key containing field values
            
        Raises:
            LLMError: If LLM invocation fails
        """
        try:
            # Format the prompt
            messages = self.prompt_template.format_messages(
                row_data=input_data.get("row_data", {}),
                fields=input_data.get("fields", {}),
                category=input_data.get("category", "")
            )
            
            # Get LLM response asynchronously
            response = await self.llm.ainvoke(messages)
            
            # Parse response
            output = self._parse_response(response.content, input_data.get("fields", {}))
            
            return {"output": output}
            
        except Exception as e:
            logger.error(f"Async LLM invocation failed: {e}")
            fields = input_data.get("fields", {})
            return {"output": {field: "Error occurred" for field in fields.keys()}}
    
    def _parse_response(self, response_content: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM response into field values.
        
        Args:
            response_content: Raw response from LLM
            fields: Expected fields dictionary
            
        Returns:
            Dictionary mapping field names to values
        """
        try:
            # Try to parse as JSON first
            parsed = json.loads(response_content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logger.debug(f"Response is not valid JSON, using fallback parsing")
        
        # Fallback: return the response for all fields
        # In a production system, you might want more sophisticated parsing
        return {field: response_content for field in fields.keys()}


class VectorStoreLLMChain(BaseLLMChain):
    """
    Enhanced chain that combines vector store retrieval with LLM processing.
    
    This chain first searches a vector store for relevant context, then
    combines that context with the row data for LLM processing.
    """
    
    def __init__(self, 
                 vector_store,  # Keep generic for now, will be typed later
                 llm_chain: LLMChain,
                 max_context_chunks: int = 5):
        """
        Initialize the Vector Store + LLM chain.
        
        Args:
            vector_store: Vector store instance for context retrieval
            llm_chain: LLMChain to use for final processing
            max_context_chunks: Maximum number of context chunks to retrieve
        """
        self.vector_store = vector_store
        self.llm_chain = llm_chain
        self.max_context_chunks = max_context_chunks
        
        # Create enhanced prompt template that includes vector context
        self.enhanced_prompt = self._create_enhanced_prompt()
        
        # Replace the LLM chain's prompt with our enhanced version
        original_llm = self.llm_chain.llm
        self.llm_chain = LLMChain(original_llm, self.enhanced_prompt)
    
    def _create_enhanced_prompt(self) -> ChatPromptTemplate:
        """Create prompt template that includes vector store context."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a data enrichment specialist with access to additional context from a knowledge base.

Company Data: {row_data}

Relevant Context from Knowledge Base:
{vector_context}

Fields to analyze: {fields}

Using both the company data and the relevant context, provide accurate, well-reasoned insights for each requested field.
Return your response as a JSON object with field names as keys and your analysis as values.

Important:
- Use the context to provide more informed responses
- Cite specific information from the context when relevant
- If context doesn't help with a field, rely on the company data
- Provide factual, concise responses"""),
            ("user", "Analyze the company data using the provided context and generate insights for the requested fields.")
        ])
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous invocation with vector store context.
        
        Args:
            input_data: Dictionary with 'row_data', 'fields', and optional 'category'
            
        Returns:
            Dictionary with 'output' key containing field values
        """
        try:
            # Get relevant context from vector store
            context = self._get_vector_context(input_data)
            
            # Add context to input data
            enhanced_input = input_data.copy()
            enhanced_input["vector_context"] = context
            
            # Process through LLM chain
            return self.llm_chain.invoke(enhanced_input)
            
        except Exception as e:
            logger.error(f"VectorStoreLLMChain invocation failed: {e}")
            # Fallback to LLM-only processing
            return self.llm_chain.invoke(input_data)
    
    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous invocation with vector store context.
        
        Args:
            input_data: Dictionary with 'row_data', 'fields', and optional 'category'
            
        Returns:
            Dictionary with 'output' key containing field values
        """
        try:
            # Get relevant context from vector store
            context = await self._get_vector_context_async(input_data)
            
            # Add context to input data
            enhanced_input = input_data.copy()
            enhanced_input["vector_context"] = context
            
            # Process through LLM chain
            return await self.llm_chain.ainvoke(enhanced_input)
            
        except Exception as e:
            logger.error(f"Async VectorStoreLLMChain invocation failed: {e}")
            # Fallback to LLM-only processing
            return await self.llm_chain.ainvoke(input_data)
    
    def _get_vector_context(self, input_data: Dict[str, Any]) -> str:
        """
        Retrieve relevant context from vector store.
        
        Args:
            input_data: Input data for building search query
            
        Returns:
            Formatted context string
        """
        try:
            # Build search query from row data
            row_data = input_data.get("row_data", {})
            query_parts = []
            
            # Add company name and description if available
            for field in ["name", "company", "description", "industry"]:
                if field in row_data and row_data[field]:
                    query_parts.append(str(row_data[field]))
            
            if not query_parts:
                return "No relevant context found."
            
            search_query = " ".join(query_parts)
            
            # Search vector store
            if hasattr(self.vector_store, 'similarity_search'):
                # Standard LangChain interface
                results = self.vector_store.similarity_search(search_query, k=self.max_context_chunks)
                context_parts = [result.page_content for result in results]
            elif hasattr(self.vector_store, 'search'):
                # Custom interface
                results = self.vector_store.search(search_query, limit=self.max_context_chunks)
                context_parts = [result.get("content", "") for result in results]
            else:
                logger.warning("Vector store doesn't have expected search methods")
                return "Vector store search not available."
            
            return "\n\n".join(context_parts) if context_parts else "No relevant context found."
            
        except Exception as e:
            logger.error(f"Vector context retrieval failed: {e}")
            return "Context retrieval failed."
    
    async def _get_vector_context_async(self, input_data: Dict[str, Any]) -> str:
        """
        Asynchronously retrieve context from vector store.
        
        For now, this is a simple wrapper around the sync version.
        In a full implementation, you'd want truly async vector store operations.
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_vector_context, input_data
        )


# Factory functions for easy chain creation
def create_simple_llm_chain(api_key: Optional[str] = None, 
                          model: str = "gpt-4o-mini", 
                          **kwargs) -> LLMChain:
    """
    Factory function to create a simple LLM chain.
    
    Args:
        api_key: OpenAI API key
        model: Model name
        **kwargs: Additional arguments
        
    Returns:
        Configured LLMChain
    """
    return LLMChain.openai(model=model, api_key=api_key, **kwargs)


def create_vector_enhanced_chain(vector_store, 
                                api_key: Optional[str] = None,
                                model: str = "gpt-4o-mini",
                                **kwargs) -> VectorStoreLLMChain:
    """
    Factory function to create a vector store enhanced chain.
    
    Args:
        vector_store: Vector store instance
        api_key: OpenAI API key  
        model: Model name
        **kwargs: Additional arguments
        
    Returns:
        Configured VectorStoreLLMChain
    """
    llm_chain = LLMChain.openai(model=model, api_key=api_key, **kwargs)
    return VectorStoreLLMChain(vector_store, llm_chain)