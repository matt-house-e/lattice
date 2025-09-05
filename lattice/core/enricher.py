"""
Main enricher class for the Lattice enrichment tool.

This is the primary interface for enriching tabular data using LLMs.
Much smaller and focused compared to the original TableEnricher class.
"""

import time
import asyncio
from typing import Dict, List, Optional, Union, AsyncGenerator, Callable
import pandas as pd
from tqdm import tqdm

from .processors import RowProcessor
from .config import EnrichmentConfig
from .exceptions import EnrichmentError, FieldValidationError, PartialEnrichmentResult
from ..data import FieldManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TableEnricher:
    """
    Main orchestrator for CSV/DataFrame enrichment.
    
    This class coordinates the enrichment process by:
    - Managing the overall workflow
    - Delegating row processing to RowProcessor
    - Handling progress tracking and reporting
    - Managing batch processing and concurrency
    - Providing both sync and async interfaces
    
    Much simpler than the original 285-line version - focuses on orchestration
    rather than doing everything itself.
    """
    
    def __init__(self, 
                 chain,  # LLMChain, VectorStoreLLMChain, etc.
                 field_manager: FieldManager,
                 config: Optional[EnrichmentConfig] = None):
        """
        Initialize the TableEnricher.
        
        Args:
            chain: The enrichment chain to use (LLMChain, VectorStoreLLMChain, etc.)
            field_manager: FieldManager instance for field definitions
            config: Configuration options (optional, uses defaults if not provided)
        """
        self.chain = chain
        self.field_manager = field_manager
        self.config = config or EnrichmentConfig()
        
        # Create the row processor
        self.processor = RowProcessor(chain, field_manager, config)
        
        logger.info(f"TableEnricher initialized with {type(chain).__name__} and {field_manager}")
    
    def enrich_dataframe(self, 
                        df: pd.DataFrame, 
                        category: str,
                        overwrite_fields: bool = None) -> pd.DataFrame:
        """
        Enrich a DataFrame by processing rows through the specified category.
        
        Args:
            df: DataFrame containing data to enrich
            category: Field category to process
            overwrite_fields: Whether to overwrite existing field values (overrides config)
            
        Returns:
            DataFrame with enriched data
            
        Raises:
            FieldValidationError: If category doesn't exist
            EnrichmentError: If enrichment process fails
        """
        if overwrite_fields is None:
            overwrite_fields = self.config.overwrite_fields
            
        logger.info(f"Starting enrichment of {len(df)} rows for category '{category}'")
        
        # Validate category exists
        if not self.field_manager.validate_category(category):
            available = ", ".join(self.field_manager.get_categories())
            raise FieldValidationError(f"Category '{category}' not found. Available: {available}")
        
        # Get fields for this category
        fields_dict = self.field_manager.get_category_fields(category)
        logger.info(f"Processing {len(fields_dict)} fields: {list(fields_dict.keys())}")
        
        # Prepare DataFrame - ensure all fields exist
        df_copy = df.copy()
        for field in fields_dict.keys():
            if field not in df_copy.columns:
                df_copy[field] = None
        
        # Process rows
        if self.config.enable_async:
            # If async is enabled, use async processing even in sync method
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an async context, can't use run()
                    logger.warning("Async processing requested but already in async context, falling back to sync")
                    return self._process_sync(df_copy, fields_dict, overwrite_fields)
                else:
                    return loop.run_until_complete(
                        self._process_async(df_copy, fields_dict, overwrite_fields)
                    )
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self._process_async(df_copy, fields_dict, overwrite_fields))
        else:
            return self._process_sync(df_copy, fields_dict, overwrite_fields)
    
    async def enrich_dataframe_async(self,
                                    df: pd.DataFrame,
                                    category: str,
                                    overwrite_fields: bool = None) -> pd.DataFrame:
        """
        Asynchronously enrich a DataFrame.
        
        Args:
            df: DataFrame containing data to enrich
            category: Field category to process
            overwrite_fields: Whether to overwrite existing field values
            
        Returns:
            DataFrame with enriched data
        """
        if overwrite_fields is None:
            overwrite_fields = self.config.overwrite_fields
            
        logger.info(f"Starting async enrichment of {len(df)} rows for category '{category}'")
        
        # Validate category
        if not self.field_manager.validate_category(category):
            available = ", ".join(self.field_manager.get_categories())
            raise FieldValidationError(f"Category '{category}' not found. Available: {available}")
        
        # Get fields and prepare DataFrame
        fields_dict = self.field_manager.get_category_fields(category)
        df_copy = df.copy()
        for field in fields_dict.keys():
            if field not in df_copy.columns:
                df_copy[field] = None
        
        return await self._process_async(df_copy, fields_dict, overwrite_fields)
    
    def _process_sync(self,
                     df: pd.DataFrame,
                     fields_dict: Dict,
                     overwrite_fields: bool) -> pd.DataFrame:
        """
        Synchronous processing of DataFrame rows.
        
        Args:
            df: DataFrame to process
            fields_dict: Fields to enrich
            overwrite_fields: Whether to overwrite existing values
            
        Returns:
            Processed DataFrame
        """
        total_rows = len(df)
        errors = []
        
        # Create progress bar if enabled
        progress_bar = None
        if self.config.enable_progress_bar:
            progress_bar = tqdm(total=total_rows, desc="Processing Rows")
        
        for idx, row in df.iterrows():
            try:
                # Check if we need to process this row
                if not self._should_process_row(row, fields_dict, overwrite_fields):
                    logger.debug(f"Skipping row {idx} - fields already populated")
                    if progress_bar:
                        progress_bar.update(1)
                    continue
                
                # Process the row
                result = self.processor.process_row(row, fields_dict)
                
                # Update the DataFrame
                self._update_row(df, idx, result, fields_dict, overwrite_fields)
                
                # Call progress callback if provided
                if self.config.progress_callback:
                    self.config.progress_callback(idx + 1, total_rows)
                
                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)
                
                # Add delay between rows if configured
                if idx < total_rows - 1 and self.config.row_delay > 0:
                    time.sleep(self.config.row_delay)
                    
            except Exception as e:
                error = EnrichmentError(f"Failed to process row: {e}", row_index=idx)
                errors.append(error)
                logger.error(f"Row {idx} processing failed: {e}")
                
                if progress_bar:
                    progress_bar.update(1)
                continue
        
        if progress_bar:
            progress_bar.close()
        
        # Log summary
        success_count = total_rows - len(errors)
        logger.info(f"Processing complete: {success_count}/{total_rows} rows successful")
        
        if errors:
            logger.warning(f"{len(errors)} rows failed processing")
            for error in errors[:5]:  # Log first 5 errors
                logger.warning(f"  Row {error.row_index}: {error.message}")
        
        return df
    
    async def _process_async(self,
                            df: pd.DataFrame,
                            fields_dict: Dict,
                            overwrite_fields: bool) -> pd.DataFrame:
        """
        Asynchronous processing of DataFrame rows with concurrency control.
        
        Args:
            df: DataFrame to process
            fields_dict: Fields to enrich
            overwrite_fields: Whether to overwrite existing values
            
        Returns:
            Processed DataFrame
        """
        total_rows = len(df)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_row_with_semaphore(idx: int, row: pd.Series) -> Optional[Dict]:
            async with semaphore:
                try:
                    if not self._should_process_row(row, fields_dict, overwrite_fields):
                        return None
                    
                    result = await self.processor.process_row_async(row, fields_dict)
                    
                    # Add delay between rows
                    if self.config.row_delay > 0:
                        await asyncio.sleep(self.config.row_delay)
                    
                    return {"idx": idx, "result": result}
                    
                except Exception as e:
                    logger.error(f"Async row {idx} processing failed: {e}")
                    return {"idx": idx, "error": e}
        
        # Process all rows concurrently
        tasks = [
            process_row_with_semaphore(idx, row)
            for idx, row in df.iterrows()
        ]
        
        # Execute with progress tracking
        completed_tasks = []
        if self.config.enable_progress_bar:
            progress_bar = tqdm(total=total_rows, desc="Processing Rows (Async)")
            
            for task in asyncio.as_completed(tasks):
                result = await task
                completed_tasks.append(result)
                progress_bar.update(1)
                
                if self.config.progress_callback:
                    self.config.progress_callback(len(completed_tasks), total_rows)
                    
            progress_bar.close()
        else:
            completed_tasks = await asyncio.gather(*tasks)
        
        # Update DataFrame with results
        errors = []
        success_count = 0
        
        for task_result in completed_tasks:
            if task_result is None:
                continue  # Skipped row
            
            if "error" in task_result:
                error = EnrichmentError(
                    f"Failed to process row: {task_result['error']}", 
                    row_index=task_result["idx"]
                )
                errors.append(error)
            else:
                self._update_row(df, task_result["idx"], task_result["result"], fields_dict, overwrite_fields)
                success_count += 1
        
        logger.info(f"Async processing complete: {success_count}/{total_rows} rows successful")
        
        if errors:
            logger.warning(f"{len(errors)} rows failed processing")
        
        return df
    
    def _should_process_row(self,
                           row: pd.Series,
                           fields_dict: Dict,
                           overwrite_fields: bool) -> bool:
        """
        Check if a row needs processing based on existing field values.
        
        Args:
            row: Row to check
            fields_dict: Fields that would be processed
            overwrite_fields: Whether to overwrite existing values
            
        Returns:
            True if row should be processed
        """
        if overwrite_fields:
            return True
        
        # Check if any fields need processing
        for field in fields_dict:
            if field in row and pd.notna(row[field]) and row[field] != "":
                continue  # Field already has value
            else:
                return True  # At least one field needs processing
        
        return False  # All fields already populated
    
    def _update_row(self,
                   df: pd.DataFrame,
                   idx: int,
                   result: Dict,
                   fields_dict: Dict,
                   overwrite_fields: bool) -> None:
        """
        Update a DataFrame row with enrichment results.
        
        Args:
            df: DataFrame to update
            idx: Row index
            result: Processing results
            fields_dict: Fields being processed
            overwrite_fields: Whether to overwrite existing values
        """
        for field in fields_dict:
            # Skip if field has value and not overwriting
            if not overwrite_fields and field in df.columns:
                if pd.notna(df.at[idx, field]) and df.at[idx, field] != "":
                    continue
            
            # Get the value from results
            value = result.get(field)
            
            # Handle different value types
            if isinstance(value, list):
                # Convert list to comma-separated string
                value = ', '.join(map(str, value))
            elif value is None:
                value = ""
            else:
                value = str(value)
            
            # Set the value
            df.at[idx, field] = value
    
    def get_enrichment_info(self) -> Dict[str, any]:
        """
        Get information about the current enrichment setup.
        
        Returns:
            Dictionary with enrichment configuration info
        """
        return {
            "chain_type": type(self.chain).__name__,
            "field_manager_info": str(self.field_manager),
            "config": {
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
                "row_delay": self.config.row_delay,
                "async_enabled": self.config.enable_async,
                "progress_bar": self.config.enable_progress_bar,
            },
            "categories": self.field_manager.get_categories(),
            "total_fields": self.field_manager.get_field_count()
        }