"""
Chain classes for the Lattice enrichment tool.

Simple, focused wrappers around LangChain functionality with clean APIs.
Provides both LLM-only chains and vector store enhanced chains.

Uses Pydantic schemas for type-safe structured outputs with validation-with-retry.
"""

import os
import json
from typing import Dict, Any, Optional, Union, TypeVar, Type

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from ..core.exceptions import LLMError, ConfigurationError
from ..schemas import StructuredResult, UsageInfo, EnrichmentResult
from ..utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


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
               enforce_json: bool = True,
               **kwargs) -> 'LLMChain':
        """
        Factory method to create an OpenAI-based LLM chain.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key (uses environment variable if not provided)
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
            enforce_json: If True, enables OpenAI JSON mode for strict JSON outputs
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
            # Respect caller-provided response_format, otherwise set JSON mode if requested
            response_format = kwargs.pop(
                "response_format",
                {"type": "json_object"} if enforce_json else None
            )
            
            if response_format is not None:
                llm = ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs
                )
            else:
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
            ("system",
            """
            
            You are a structured data enrichment engine operating over tabular rows. 
            Given one input row, a set of field specifications, and optional external context, produce a JSON object with EXACTLY the requested fields as keys and values that satisfy each field’s constraints.

            INPUTS
            - Row Data (JSON): {row_data}
            - Field Specifications (JSON): {fields}
            Each field item contains at least:
                - prompt: the concrete instruction for this field
                - instructions: format/refinement constraints
                - data_type: expected type (e.g., String, Number, Boolean, Date, JSON, List[String])
                - examples: optional examples of ideal outputs
            - Optional Context:
            - Vector Context: {vector_context}
            - Web Search Results: {web_search_results}
            If any optional context is empty or unavailable, ignore it.

            OUTPUT CONTRACT
            - Return ONLY a single valid JSON object. No prose, no code fences, no explanations.
            - Top-level keys MUST be exactly the field names present in Field Specifications.
            - Values MUST comply with each field’s data_type and instructions.
            - Keep outputs concise and information-dense. Avoid filler language.
            - If the row and context are insufficient to answer a field, return "Unable to determine" (String), null (for non-String types), or an empty list (for list types). Never fabricate sources or numbers.
            - When context includes citations or sources and the field’s instructions ask for sources, include terse source references inline (e.g., "… (Reuters, 2024)"). Do not add URLs unless clearly present in context.
            - Do not include any keys not requested. Do not include reasoning if not explicitly requested.

            DECISION GUIDELINES
            - If sources contradict, prefer the most specific and recent context; otherwise, prefer Row Data.
            - Follow examples to style the answer when provided, but never copy them verbatim.
            - Normalize simple formatting:
            - Numbers: plain numerals; include units only if requested.
            - Dates: ISO-8601 (YYYY-MM-DD) unless instructions specify another format.
            - Lists: small, ordered by relevance; 4 items max unless otherwise stated.

            EXECUTION
            For each field:
            1) Read prompt and instructions.
            2) Check Row Data; then consult Vector/Web context if helpful.
            3) Produce an accurate value that satisfies data_type and instructions.
            4) If insufficient evidence, use the fallback policy above without guessing.

            Return ONLY the final JSON object with the requested fields as keys.
            
            """),
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
                category=input_data.get("category", ""),
                vector_context=input_data.get("vector_context", ""),
                web_search_results=input_data.get("web_search_results", "")
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
                category=input_data.get("category", ""),
                vector_context=input_data.get("vector_context", ""),
                web_search_results=input_data.get("web_search_results", "")
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

    def complete_structured(
        self,
        input_data: Dict[str, Any],
        schema: Type[T] = EnrichmentResult,
        max_retries: int = 2
    ) -> StructuredResult[T]:
        """
        Synchronous structured completion with Pydantic validation.

        Uses OpenAI JSON mode and validates response against Pydantic schema.
        On validation error, sends error details back to LLM for self-correction.

        Args:
            input_data: Dictionary with 'row_data', 'fields', and optional 'category'
            schema: Pydantic model class for output validation (default: EnrichmentResult)
            max_retries: Number of retry attempts on validation failure

        Returns:
            StructuredResult[T] with validated data and usage info

        Raises:
            LLMError: If validation fails after all retries
        """
        # Build messages
        messages = self.prompt_template.format_messages(
            row_data=input_data.get("row_data", {}),
            fields=input_data.get("fields", {}),
            category=input_data.get("category", ""),
            vector_context=input_data.get("vector_context", ""),
            web_search_results=input_data.get("web_search_results", "")
        )

        # Inject JSON schema into system prompt
        json_schema = schema.model_json_schema()
        schema_instruction = f"You MUST respond with valid JSON matching this schema: {json.dumps(json_schema)}\n\n"

        # Prepend schema to first message content
        if messages and hasattr(messages[0], 'content'):
            messages[0].content = schema_instruction + messages[0].content

        # Capture full prompt for debugging
        full_prompt = "\n\n".join(
            f"[{getattr(msg, 'type', 'unknown').upper()}]\n{msg.content}"
            for msg in messages
        )

        # Retry loop with validation feedback
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                content = response.content

                # Parse JSON
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError as e:
                    if attempt < max_retries:
                        logger.warning(f"Invalid JSON (attempt {attempt + 1}): {e}")
                        messages.append(self._create_retry_message(
                            f"Invalid JSON: {str(e)}. Please respond with valid JSON only."
                        ))
                        continue
                    raise LLMError(f"Failed to get valid JSON after {max_retries + 1} attempts")

                # Validate with Pydantic
                validated = schema.model_validate(result_data)

                # Extract usage info
                usage = self._extract_usage(response)

                return StructuredResult[schema](
                    data=validated,
                    usage=usage,
                    full_prompt=full_prompt
                )

            except ValidationError as e:
                last_error = e
                if attempt < max_retries:
                    error_details = e.json()
                    logger.warning(f"Validation error (attempt {attempt + 1}): {error_details}")
                    messages.append(self._create_retry_message(
                        f"Validation failed: {error_details}. Please fix these issues."
                    ))
                    continue
                raise LLMError(
                    f"Validation failed after {max_retries + 1} attempts: {last_error}"
                )

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Error (attempt {attempt + 1}): {e}")
                    continue
                raise LLMError(f"Structured completion failed: {e}")

        raise LLMError(f"Structured completion failed after {max_retries + 1} attempts")

    async def acomplete_structured(
        self,
        input_data: Dict[str, Any],
        schema: Type[T] = EnrichmentResult,
        max_retries: int = 2
    ) -> StructuredResult[T]:
        """
        Asynchronous structured completion with Pydantic validation.

        Uses OpenAI JSON mode and validates response against Pydantic schema.
        On validation error, sends error details back to LLM for self-correction.

        Args:
            input_data: Dictionary with 'row_data', 'fields', and optional 'category'
            schema: Pydantic model class for output validation (default: EnrichmentResult)
            max_retries: Number of retry attempts on validation failure

        Returns:
            StructuredResult[T] with validated data and usage info

        Raises:
            LLMError: If validation fails after all retries
        """
        # Build messages
        messages = self.prompt_template.format_messages(
            row_data=input_data.get("row_data", {}),
            fields=input_data.get("fields", {}),
            category=input_data.get("category", ""),
            vector_context=input_data.get("vector_context", ""),
            web_search_results=input_data.get("web_search_results", "")
        )

        # Inject JSON schema into system prompt
        json_schema = schema.model_json_schema()
        schema_instruction = f"You MUST respond with valid JSON matching this schema: {json.dumps(json_schema)}\n\n"

        # Prepend schema to first message content
        if messages and hasattr(messages[0], 'content'):
            messages[0].content = schema_instruction + messages[0].content

        # Capture full prompt for debugging
        full_prompt = "\n\n".join(
            f"[{getattr(msg, 'type', 'unknown').upper()}]\n{msg.content}"
            for msg in messages
        )

        # Retry loop with validation feedback
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await self.llm.ainvoke(messages)
                content = response.content

                # Parse JSON
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError as e:
                    if attempt < max_retries:
                        logger.warning(f"Invalid JSON (attempt {attempt + 1}): {e}")
                        messages.append(self._create_retry_message(
                            f"Invalid JSON: {str(e)}. Please respond with valid JSON only."
                        ))
                        continue
                    raise LLMError(f"Failed to get valid JSON after {max_retries + 1} attempts")

                # Validate with Pydantic
                validated = schema.model_validate(result_data)

                # Extract usage info
                usage = self._extract_usage(response)

                return StructuredResult[schema](
                    data=validated,
                    usage=usage,
                    full_prompt=full_prompt
                )

            except ValidationError as e:
                last_error = e
                if attempt < max_retries:
                    error_details = e.json()
                    logger.warning(f"Async validation error (attempt {attempt + 1}): {error_details}")
                    messages.append(self._create_retry_message(
                        f"Validation failed: {error_details}. Please fix these issues."
                    ))
                    continue
                raise LLMError(
                    f"Validation failed after {max_retries + 1} attempts: {last_error}"
                )

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Async error (attempt {attempt + 1}): {e}")
                    continue
                raise LLMError(f"Async structured completion failed: {e}")

        raise LLMError(f"Async structured completion failed after {max_retries + 1} attempts")

    def _create_retry_message(self, error_text: str):
        """Create a retry message for validation feedback."""
        from langchain_core.messages import HumanMessage
        return HumanMessage(content=error_text)

    def _extract_usage(self, response) -> UsageInfo:
        """Extract token usage from LLM response."""
        # Try to get usage from response metadata
        usage_data = getattr(response, 'usage_metadata', None)
        if usage_data:
            return UsageInfo(
                prompt_tokens=getattr(usage_data, 'input_tokens', 0),
                completion_tokens=getattr(usage_data, 'output_tokens', 0)
            )

        # Try response_metadata (LangChain format)
        response_meta = getattr(response, 'response_metadata', {})
        if 'token_usage' in response_meta:
            token_usage = response_meta['token_usage']
            return UsageInfo(
                prompt_tokens=token_usage.get('prompt_tokens', 0),
                completion_tokens=token_usage.get('completion_tokens', 0)
            )

        # Default if no usage info available
        return UsageInfo(prompt_tokens=0, completion_tokens=0)

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
            # Try to parse as raw JSON first
            parsed = json.loads(response_content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logger.debug(f"Response is not raw JSON, attempting markdown extraction")
        
        # Try to extract JSON from markdown code blocks
        try:
            extracted_json = self._extract_json_from_markdown(response_content)
            if extracted_json:
                parsed = json.loads(extracted_json)
                if isinstance(parsed, dict):
                    return parsed
        except json.JSONDecodeError:
            logger.debug(f"Extracted content is not valid JSON")
        except Exception as e:
            logger.debug(f"Error during markdown JSON extraction: {e}")
        
        # Enhanced fallback: try to extract individual field values from response
        fallback_result = self._extract_fields_from_text(response_content, fields)
        if fallback_result:
            return fallback_result
        
        # Final fallback: return "Unable to parse" for all fields instead of duplicating content
        logger.warning(f"Could not parse response, returning default values for all fields")
        return {field: "Unable to parse response" for field in fields.keys()}
    
    def _extract_json_from_markdown(self, text: str) -> Optional[str]:
        """
        Extract JSON content from markdown code blocks.
        
        Args:
            text: Text that may contain markdown-wrapped JSON
            
        Returns:
            Extracted JSON string or None if not found
        """
        import re
        
        # Pattern to match JSON code blocks (```json ... ```)
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Try generic code blocks (``` ... ```)
        generic_pattern = r'```\s*(.*?)\s*```'
        match = re.search(generic_pattern, text, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            # Check if it looks like JSON (starts with { and ends with })
            if content.strip().startswith('{') and content.strip().endswith('}'):
                return content
        
        # Try to find JSON-like content without code blocks
        json_like_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_like_pattern, text, re.DOTALL)
        
        if match:
            return match.group(0)
        
        return None
    
    def _extract_fields_from_text(self, text: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract field values from unstructured text.
        
        Args:
            text: Response text to parse
            fields: Expected fields dictionary
            
        Returns:
            Dictionary with extracted field values or None if unsuccessful
        """
        import re
        
        result = {}
        field_names = list(fields.keys())
        
        # Try to find field: value patterns in the text
        for field_name in field_names:
            # Look for patterns like "field_name": "value" or field_name: value
            patterns = [
                rf'"{field_name}"\s*:\s*"([^"]*)"',
                rf'"{field_name}"\s*:\s*([^,\n}}]+)',
                rf'{field_name}\s*:\s*"([^"]*)"',
                rf'{field_name}\s*:\s*([^,\n}}]+)',
                rf'{field_name}\s+is\s*"([^"]*)"',  # Handle "field is 'value'" pattern
                rf'{field_name}\s+should\s+be\s*"([^"]*)"',  # Handle "field should be 'value'" pattern
                rf'{field_name}.*?"([^"]*)"',  # General pattern with quotes
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Clean up common artifacts
                    value = value.strip(',').strip()
                    if value and value not in ['null', 'None', '']:
                        result[field_name] = value
                        break
        
        # Only return result if we found at least one field
        return result if result else None


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
            ("system", """You are a data enrichment specialist

Row Data: {row_data}

Relevant Context from Knowledge Base:
{vector_context}

Fields to analyze: {fields}

Using both the row data and the relevant context, provide accurate, well-reasoned insights for each requested field.
Return your response as a JSON object with field names as keys and your analysis as values.

Important:
- Use the context to provide more informed responses
- Cite specific information from the context when relevant
- If context doesn't help with a field, rely on the company data
- Provide factual, concise responses"""),
            ("user", "Analyze the row data using the provided context and generate insights for the requested fields.")
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