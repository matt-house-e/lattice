"""
Custom exceptions for the Lattice enrichment tool.

Provides specific exception types for different failure modes
with helpful error messages and context.
"""

from typing import Optional, List, Dict, Any


class EnrichmentError(Exception):
    """Base exception for all enrichment-related errors."""
    
    def __init__(self, message: str, row_index: Optional[int] = None, field: Optional[str] = None):
        self.message = message
        self.row_index = row_index
        self.field = field
        
        # Build descriptive error message
        error_parts = [message]
        if row_index is not None:
            error_parts.append(f"Row: {row_index}")
        if field is not None:
            error_parts.append(f"Field: {field}")
            
        super().__init__(" | ".join(error_parts))


class FieldValidationError(EnrichmentError):
    """Raised when field definitions are invalid or missing."""
    pass


class LLMError(EnrichmentError):
    """Raised when LLM processing fails."""
    pass


class VectorStoreError(EnrichmentError):
    """Raised when vector store operations fail."""
    pass


class ConfigurationError(EnrichmentError):
    """Raised when configuration is invalid."""
    pass


class PartialEnrichmentResult:
    """
    Container for enrichment results that may have some failures.
    
    Allows graceful handling of partial successes where some rows
    were enriched successfully but others failed.
    """
    
    def __init__(self, successful_rows: List[Dict[str, Any]], errors: List[EnrichmentError]):
        self.successful_rows = successful_rows
        self.errors = errors
        self.total_rows = len(successful_rows) + len(errors)
        self.success_rate = len(successful_rows) / self.total_rows if self.total_rows > 0 else 0.0
    
    @property
    def has_errors(self) -> bool:
        """True if any rows failed to process."""
        return len(self.errors) > 0
    
    @property
    def is_complete_success(self) -> bool:
        """True if all rows processed successfully."""
        return len(self.errors) == 0
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error types."""
        error_counts = {}
        for error in self.errors:
            error_type = type(error).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts
    
    def __str__(self) -> str:
        return f"PartialEnrichmentResult({len(self.successful_rows)} success, {len(self.errors)} errors, {self.success_rate:.1%} success rate)"