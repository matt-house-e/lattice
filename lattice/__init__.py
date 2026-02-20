"""Lattice — Enrichment Pipeline Engine.

A programmatic enrichment engine built on composable, column-oriented
pipeline steps with Pydantic validation, checkpointing, and async concurrency.
"""

# Set up default logging when package is imported
from .utils.logger import setup_logging
setup_logging(level="WARNING", format_type="console", include_timestamp=False)

# Public API
from .core import Enricher, EnrichmentConfig, EnrichmentError, FieldValidationError, RowError
from .pipeline import Pipeline, PipelineResult
from .steps import Step, StepContext, StepResult, FunctionStep, LLMStep
from .schemas.base import CostSummary
from .schemas.field_spec import FieldSpec

__version__ = "0.4.0"
__author__ = "Lattice Team"

__all__ = [
    # Primary API
    "Pipeline",
    "PipelineResult",
    "LLMStep",
    "FunctionStep",
    "EnrichmentConfig",
    # Step protocol
    "Step",
    "StepContext",
    "StepResult",
    # Schemas
    "FieldSpec",
    # Results & errors
    "CostSummary",
    "RowError",
    "EnrichmentError",
    "FieldValidationError",
    # Internal runner (power users)
    "Enricher",
]
