"""
Row processing logic for the Lattice enrichment tool.

Contains the core logic for processing individual rows through enrichment chains.
This is extracted from the original TableEnricher to follow single responsibility principle.
"""

from typing import Dict, Any, Optional, Union
import pandas as pd
import time
import asyncio

from .exceptions import EnrichmentError, LLMError
from .config import EnrichmentConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RowProcessor:
    """
    Handles processing of individual rows through enrichment chains.
    
    This class is responsible for:
    - Taking a row of data and field specifications
    - Formatting the data for the enrichment chain
    - Invoking the chain and handling responses
    - Converting responses back to the expected format
    - Error handling and retries for individual rows
    """
    
    def __init__(self, chain, field_manager, config: Optional[EnrichmentConfig] = None):
        """
        Initialize the RowProcessor.
        
        Args:
            chain: The enrichment chain to use (LLMChain, VectorStoreLLMChain, etc.)
            field_manager: FieldManager instance for field specifications
            config: Configuration options (optional)
        """
        self.chain = chain
        self.field_manager = field_manager
        self.config = config or EnrichmentConfig()
        
    def process_row(self, row: pd.Series, fields_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a single row using the enrichment chain.
        
        Args:
            row: DataFrame row to process (as pandas Series)
            fields_dict: Dictionary of fields to enrich with their specifications
            
        Returns:
            Dictionary containing processed values for each field
            
        Raises:
            EnrichmentError: If processing fails after all retries
        """
        row_index = getattr(row, 'name', None)  # Get row index if available
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._attempt_process_row(row, fields_dict)
            except Exception as e:
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"Row processing attempt {attempt + 1} failed (row {row_index}): {e}. "
                        f"Retrying in {self.config.retry_delay}s..."
                    )
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    # Final attempt failed
                    logger.error(f"Row processing failed after {self.config.max_retries + 1} attempts (row {row_index}): {e}")
                    raise LLMError(f"Failed to process row after {self.config.max_retries + 1} attempts: {e}", row_index=row_index)
    
    async def process_row_async(self, row: pd.Series, fields_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a single row asynchronously.
        
        Args:
            row: DataFrame row to process (as pandas Series)
            fields_dict: Dictionary of fields to enrich with their specifications
            
        Returns:
            Dictionary containing processed values for each field
            
        Raises:
            EnrichmentError: If processing fails after all retries
        """
        row_index = getattr(row, 'name', None)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._attempt_process_row_async(row, fields_dict)
            except Exception as e:
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"Async row processing attempt {attempt + 1} failed (row {row_index}): {e}. "
                        f"Retrying in {self.config.retry_delay}s..."
                    )
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    logger.error(f"Async row processing failed after {self.config.max_retries + 1} attempts (row {row_index}): {e}")
                    raise LLMError(f"Failed to process row after {self.config.max_retries + 1} attempts: {e}", row_index=row_index)
    
    def _attempt_process_row(self, row: pd.Series, fields_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Single attempt to process a row (extracted from original enrichment.py).
        
        Args:
            row: DataFrame row to process
            fields_dict: Dictionary of fields to enrich
            
        Returns:
            Dictionary containing processed values for each field
        """
        try:
            # Prepare row data for processing (exclude the fields we're trying to enrich)
            row_data = row.drop(list(fields_dict.keys()), errors='ignore')
            
            # Convert Timestamp objects to strings for serialization
            row_data = row_data.apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)
            
            # Get category for these fields (for context)
            category = None
            if self.field_manager:
                try:
                    # Get category from first field
                    first_field = next(iter(fields_dict))
                    category = self.field_manager.get_field_category(first_field)
                except Exception as e:
                    logger.warning(f"Failed to get category for fields: {e}")

            # Prepare input for the chain
            chain_input = {
                "row_data": row_data.to_dict(),
                "fields": fields_dict,
                "category": category
            }
            
            # Process through enrichment chain
            if hasattr(self.chain, 'invoke'):
                response = self.chain.invoke(chain_input)
            else:
                # Fallback for chains that don't follow LangChain interface
                response = self.chain(chain_input)
            
            # Extract field values from response
            if isinstance(response, dict) and "output" in response:
                field_values = response["output"]
            elif isinstance(response, dict):
                field_values = response
            else:
                # If response is not a dict, try to parse it
                logger.warning(f"Unexpected response type: {type(response)}, attempting to parse")
                field_values = self._parse_response(response, fields_dict)
            
            # Ensure we have values for all requested fields
            result = {}
            for field in fields_dict:
                if field in field_values:
                    result[field] = field_values[field]
                else:
                    logger.warning(f"Missing field '{field}' in chain response")
                    result[field] = None
                    
            return result
            
        except Exception as e:
            logger.error(f"Error in _attempt_process_row: {e}")
            # Return None/empty values for all fields on error
            return {field: None for field in fields_dict}
    
    async def _attempt_process_row_async(self, row: pd.Series, fields_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Async version of _attempt_process_row.
        
        Args:
            row: DataFrame row to process
            fields_dict: Dictionary of fields to enrich
            
        Returns:
            Dictionary containing processed values for each field
        """
        try:
            # Prepare row data (same as sync version)
            row_data = row.drop(list(fields_dict.keys()), errors='ignore')
            row_data = row_data.apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)
            
            category = None
            if self.field_manager:
                try:
                    first_field = next(iter(fields_dict))
                    category = self.field_manager.get_field_category(first_field)
                except Exception as e:
                    logger.warning(f"Failed to get category for fields: {e}")

            chain_input = {
                "row_data": row_data.to_dict(),
                "fields": fields_dict,
                "category": category
            }
            
            # Try async invocation first, fall back to sync
            if hasattr(self.chain, 'ainvoke'):
                response = await self.chain.ainvoke(chain_input)
            elif hasattr(self.chain, 'invoke'):
                # Run sync method in thread pool to avoid blocking
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.chain.invoke, chain_input
                )
            else:
                # Fallback for non-LangChain chains
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.chain, chain_input
                )
            
            # Extract field values (same logic as sync version)
            if isinstance(response, dict) and "output" in response:
                field_values = response["output"]
            elif isinstance(response, dict):
                field_values = response
            else:
                field_values = self._parse_response(response, fields_dict)
            
            result = {}
            for field in fields_dict:
                if field in field_values:
                    result[field] = field_values[field]
                else:
                    logger.warning(f"Missing field '{field}' in chain response")
                    result[field] = None
                    
            return result
            
        except Exception as e:
            logger.error(f"Error in _attempt_process_row_async: {e}")
            return {field: None for field in fields_dict}
    
    def _parse_response(self, response: Any, fields_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse non-standard responses into field values.
        
        Args:
            response: The response from the chain
            fields_dict: Expected fields
            
        Returns:
            Dictionary of field values
        """
        # Try to parse as string (maybe JSON)
        if isinstance(response, str):
            try:
                import json
                parsed = json.loads(response)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # If we can't parse it, return the response for all fields
        logger.warning(f"Could not parse response: {response}")
        return {field: str(response) for field in fields_dict}
    
    def get_processing_info(self) -> Dict[str, Any]:
        """
        Get information about the current processor configuration.
        
        Returns:
            Dictionary with processor information
        """
        return {
            "chain_type": type(self.chain).__name__,
            "field_manager": str(self.field_manager) if self.field_manager else None,
            "max_retries": self.config.max_retries,
            "retry_delay": self.config.retry_delay,
            "async_enabled": self.config.enable_async
        }