"""
Custom exceptions for the Lattice enrichment tool.

Provides specific exception types for different failure modes
with helpful error messages and context.
"""

from dataclasses import dataclass, field
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


class ConfigurationError(EnrichmentError):
    """Raised when configuration is invalid."""
    pass


class StepError(EnrichmentError):
    """Raised when a pipeline step fails."""

    def __init__(self, message: str, step_name: Optional[str] = None, **kwargs: Any):
        self.step_name = step_name
        super().__init__(message, **kwargs)


class PipelineError(EnrichmentError):
    """Raised when pipeline construction or execution fails."""
    pass


@dataclass
class RowError:
    """Per-row failure record for partial result tracking."""

    row_index: int
    step_name: str
    error: BaseException
    error_type: str = ""

    def __post_init__(self) -> None:
        if not self.error_type:
            self.error_type = type(self.error).__name__

    def __str__(self) -> str:
        return f"RowError(row={self.row_index}, step='{self.step_name}', {self.error_type}: {self.error})"
